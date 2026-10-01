"""Figure 2 input: exactly 100,000 accepted inputs per model and mixing case (800,000 total).

For WR, Brusselator, Oregonator and CS14, half ideal (Gamma = 0) and half nonideal
(random Gamma > 0), rates are drawn by crn_cost.sample and each input is integrated
once with the numba engine of crn_cost.py; the four terms of the thermodynamic-metric
bound |lambda_1 + Delta_1| <= sqrt(sigma_ps B_1), i.e. lambda_1, Delta_1, B_1 and
sigma_ps (plus sigma), are computed separately along the leading tangent direction
and written to data_fig2/fig_2_inputs.csv.

Numerical solving happens only in this script. The two Figure 2 plotters read CSV.
All attempts are committed to SQLite; final selection is by proposal index, not
completion time. A partial run never masquerades as the complete production CSV.
"""
import os
for _v in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(_v,'1')
import csv,json,math,sqlite3,time,argparse,hashlib
from pathlib import Path
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import get_context
from crn_cost import Config,VERSION,run_point,sample,warmup

MODELS=('wr','brusselator','oregonator','cs14')
TARGET=100000         # accepted inputs per model and per mixing case (ideal / nonideal)
TERMS=('lambda1','Delta1','B1','sigma_ps')   # the four independently computed terms of |lambda+Delta| <= sqrt(sigma_ps B)


def save_json(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.%d.tmp'%os.getpid())
    t.write_text(json.dumps(obj,indent=2,allow_nan=False));t.replace(p)

SAMPLER_VERSION='fig2-exact-2-numba'


def parser(description):
    p=argparse.ArgumentParser(description=description,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output',default='data_fig2');p.add_argument('--workers',type=int,default=1)
    p.add_argument('--seed',type=int,default=20260927)
    p.add_argument('--preset',choices=['production','smoke'],default='production')
    for name in ('transient','duration','rtol','atol'):p.add_argument('--'+name,type=float)
    p.add_argument('--blocks',type=int);p.add_argument('--no-confirmation',action='store_true')
    return p


def config(args):
    c=Config()
    if args.preset=='smoke':c.transient=80.;c.duration=120.;c.confirmation=False
    for key in ('transient','duration','rtol','atol','blocks'):
        if getattr(args,key) is not None:setattr(c,key,getattr(args,key))
    if args.no_confirmation:c.confirmation=False
    if min(c.transient,c.duration,c.rtol,c.atol)<=0 or c.blocks<2:raise ValueError('Positive time/tolerance settings and at least two blocks required')
    return c


def job_id(p,c):
    raw=json.dumps([VERSION,p,asdict(c)],sort_keys=True,separators=(',',':'))
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def flatten(r):
    """One CSV row: results, full input (rates, reservoir chemical potentials, x0, d0, Gamma), settings."""
    p=r['parameters'];row={k:v for k,v in r.items() if k not in ('parameters','config') and not isinstance(v,(dict,list))}
    row.update(model=p['model'],nonideal=p['nonideal'],seed=p.get('seed'),index=p.get('index'),branch=p.get('branch'),time_unit=p['time_unit'],drive=p.get('drive'))
    for i,(kp,km) in enumerate(zip(p['k'][::2],p['k'][1::2]),1):
        row.update({'k_plus_'+str(i):kp,'k_minus_'+str(i):km,'mu_F_'+str(i):math.log(kp/km),'mu_W_'+str(i):0.})
    for key in ('x0','d0'):row.update({key+'_'+str(i+1):x for i,x in enumerate(p[key])})
    for i,line in enumerate(p['Gamma']):
        for j,g in enumerate(line):row['Gamma_'+str(i+1)+'_'+str(j+1)]=g
    for key in ('spectrum','xmin','xmax','endpoint','xstar','xcycle','floquet_moduli','eigen_real'):
        row.update({key+'_'+str(i+1):x for i,x in enumerate(r.get(key) or [])})
    row.update({'cfg_'+k:v for k,v in r['config'].items()})
    row['parameters_json']=json.dumps(p,separators=(',',':'),allow_nan=False)
    for key in ('spectrum_blocks','lambda_H_blocks'):
        if key in r:row[key+'_json']=json.dumps(r[key],separators=(',',':'))
    return row
CASES=tuple((m,n) for m in MODELS for n in (False,True))


def calculate(task):
    model,nonideal,index,seed,cfgdict=task
    p=sample(model,index,seed,nonideal);c=Config(**cfgdict)
    result=run_point(p,c);result['point_id']=job_id(p,c)
    return result


def accepted(result):
    if result.get('status')!='ok' or not result.get('selected'):return False,'numerical_or_classification_check'
    if result.get('cls') not in ('fixed','cycle','chaos_candidate'):return False,'unresolved_class'
    values=[result.get(k) for k in ('lambda_H','Delta1','B1','sigma_ps','sigma')]
    if any(v is None or not math.isfinite(v) for v in values):return False,'nonfinite_term'
    if min(values[1:])<=0:return False,'nonpositive_plot_term'
    try:cost=values[1]**2/values[2]
    except (OverflowError,ZeroDivisionError):return False,'undefined_motion_cost'
    if not math.isfinite(cost) or cost<=0:return False,'undefined_motion_cost'
    # Deliberately no comparison of either side of the thermodynamic inequality.
    return True,''

# ---------------------------------------------------------------- layout -----
# <output>/spec.json                     run settings; every worker process checks it
# <output>/claims/<case>/<block>/         one directory per claimed block of BLOCK proposal indices
#                                          (mkdir is atomic, also on NFS: exactly one claimant)
# <output>/workers/<name>.sqlite          every attempt of one worker process (one per node)
# <output>/progress/<name>.json           heartbeat and counts of that worker (read by the others)
#
# Any number of worker processes on any number of nodes, with any core counts, cooperate:
# each claims the next free block of a case that still needs inputs, so fast nodes simply
# do more blocks. The selection (first `target` accepted inputs per case in proposal-index
# order, within the contiguous range of completed indices) is independent of who computed
# what, and identical to a one-process run with the same seed.

BLOCK=32


def case_name(c):return c[0]+'_'+('nonideal' if c[1] else 'ideal')


def init_run(output,spec):
    """Create the run directory or check that it was created with the same settings."""
    out=Path(output);out.mkdir(parents=True,exist_ok=True);f=out/'spec.json'
    if f.exists():
        stored=json.loads(f.read_text())
        if stored!=json.loads(json.dumps(spec)):
            raise ValueError('%s was created with other settings (seed, target, numerical options). Use the same options or another --output.'%f)
        return stored
    for c in CASES:(out/'claims'/case_name(c)).mkdir(parents=True,exist_ok=True)
    (out/'workers').mkdir(exist_ok=True);(out/'progress').mkdir(exist_ok=True)
    save_json(f,spec);return spec


def open_db(path,spec,readonly=False):
    path=Path(path)
    if readonly:
        db=sqlite3.connect('file:%s?mode=ro'%path,uri=True,timeout=600)
    else:
        db=sqlite3.connect(str(path),timeout=600)
        db.execute('PRAGMA journal_mode=DELETE');db.execute('PRAGMA synchronous=NORMAL')   # safe on NFS/Lustre
        db.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY,value TEXT NOT NULL)')
        db.execute('''CREATE TABLE IF NOT EXISTS attempts (model TEXT NOT NULL,nonideal INTEGER NOT NULL,
            sample_index INTEGER NOT NULL,accepted INTEGER NOT NULL,reason TEXT NOT NULL,result TEXT NOT NULL,
            PRIMARY KEY (model,nonideal,sample_index))''')
        if not db.execute("SELECT 1 FROM metadata WHERE key='spec'").fetchone():
            db.execute('INSERT INTO metadata VALUES (?,?)',('spec',json.dumps(spec,sort_keys=True)));db.commit()
    stored=json.loads(db.execute("SELECT value FROM metadata WHERE key='spec'").fetchone()[0])
    if stored!=json.loads(json.dumps(spec)):
        db.close();raise ValueError('%s holds a run with other settings'%path)
    return db


def indices(db):
    done={c:set() for c in CASES};good={c:set() for c in CASES}
    for m,n,i,a in db.execute('SELECT model,nonideal,sample_index,accepted FROM attempts'):
        done[m,bool(n)].add(i)
        if a:good[m,bool(n)].add(i)
    return done,good


def union(output,spec):
    """Completed and accepted proposal indices of all workers (read only)."""
    done={c:set() for c in CASES};good={c:set() for c in CASES};where={c:{} for c in CASES};dbs=[]
    for p in sorted((Path(output)/'workers').glob('*.sqlite')):
        db=open_db(p,spec,readonly=True);dbs.append(db);d,g=indices(db)
        for c in CASES:
            done[c]|=d[c];good[c]|=g[c]
            for i in g[c]:where[c].setdefault(i,db)
    return done,good,where,dbs


def prefix(done_c):
    i=0
    while i in done_c:i+=1
    return i


def alive(output,owner,stale_after):
    """A worker is alive while its heartbeat file is younger than stale_after seconds."""
    f=Path(output)/'progress'/(owner+'.json')
    try:return time.time()-f.stat().st_mtime<stale_after
    except OSError:return False


def release_stale(output,spec,stale_after=None):
    """Free the claims of unfinished blocks whose worker is gone (no heartbeat for stale_after
    seconds), so that another worker redoes them. stale_after=None frees every unfinished block
    (only when no worker runs). Redoing a block twice is harmless: inputs are deterministic."""
    done,_,_,dbs=union(output,spec);n=0
    for db in dbs:db.close()
    for c in CASES:
        for d in (Path(output)/'claims'/case_name(c)).iterdir():
            b=int(d.name)
            if all(i in done[c] for i in range(b*BLOCK,(b+1)*BLOCK)):continue
            if stale_after is not None:
                try:owner=(d/'owner').read_text().strip()
                except OSError:owner=None
                if owner and alive(output,owner,stale_after):continue
                if not owner and time.time()-d.stat().st_mtime<stale_after:continue   # just being claimed
            try:
                (d/'owner').unlink()
            except OSError:pass
            try:d.rmdir();n+=1
            except OSError:pass
    return n


# --------------------------------------------------------------- sampling ----
def generate(spec,output,name,workers=1,run_seconds=None,margin=3.,worker=calculate):
    """One worker process (typically one per node) with `workers` cores. Stops when every case has
    enough claimed blocks for `target` accepted inputs (acceptance estimated from all workers)."""
    import signal,random
    from collections import deque
    from concurrent.futures import wait,FIRST_COMPLETED
    out=Path(output);target=spec['target_per_case']
    db=open_db(out/'workers'/(name+'.sqlite'),spec);done,good=indices(db)
    stop={'signal':None}
    def handler(signum,frame):stop['signal']=signum
    signal.signal(signal.SIGTERM,handler)
    warmup()
    pool=ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn'),initializer=warmup) if workers>1 else None
    queue=deque();inflight={};blocks={c:set() for c in CASES};pending=0;start=time.monotonic()
    glob={'p':{c:0.5 for c in CASES},'att':{c:0 for c in CASES},'acc':{c:0 for c in CASES},'t':-1e9};rr=[0];last_commit=[time.monotonic()]
    def publish():
        save_json(out/'progress'/(name+'.json'),dict(name=name,time=time.time(),workers=workers,
            elapsed_s=round(time.monotonic()-start),cases={case_name(c):dict(attempted=len(done[c]),accepted=len(good[c])) for c in CASES}))
    def refresh():
        """Global acceptance estimate from every worker's heartbeat file."""
        if time.monotonic()-glob['t']<20:
            return
        att={c:len(done[c]) for c in CASES};acc={c:len(good[c]) for c in CASES}   # own counts live
        for f in (out/'progress').glob('*.json'):
            if f.stem==name:continue
            try:d=json.loads(f.read_text())
            except (ValueError,OSError):continue
            for c in CASES:
                e=d['cases'].get(case_name(c),{});att[c]+=e.get('attempted',0);acc[c]+=e.get('accepted',0)
        glob['p']={c:max(0.02,(acc[c]+1)/(att[c]+2)) for c in CASES};glob['att']=att;glob['acc']=acc;glob['t']=time.monotonic()
    def claimed(c):return [int(d.name) for d in (out/'claims'/case_name(c)).iterdir()]
    def wanted(c,cl):
        # expected accepted inputs once every claimed block is done: accepted so far (all workers)
        # plus the acceptance rate times the claimed indices not yet computed
        expect=glob['acc'][c]+glob['p'][c]*max(0,len(cl)*BLOCK-glob['att'][c])
        return expect<target+margin*math.sqrt(target)
    def claim(retry=True):
        """Claim the lowest free block of the next case that still needs inputs."""
        refresh()
        for _ in range(len(CASES)):
            c=CASES[rr[0]%len(CASES)];rr[0]+=1;cl=claimed(c)
            if not wanted(c,cl):continue
            taken=set(cl);b=0
            while True:
                while b in taken:b+=1
                try:
                    bd=out/'claims'/case_name(c)/('%07d'%b);bd.mkdir();(bd/'owner').write_text(name);break
                except FileExistsError:taken.add(b)
            blocks[c].add(b)
            for i in range(b*BLOCK,(b+1)*BLOCK):
                if i not in done[c]:queue.append((c[0],c[1],i,spec['seed'],spec['config']))
            return True
        if retry:                                   # before giving up, look again with fresh counts
            glob['t']=-1e9;return claim(False)
        return False
    def record(task,r):
        nonlocal pending
        m,n,idx,_,_=task;ok,reason=accepted(r)
        if (r['parameters']['model'],bool(r['parameters']['nonideal']),r['parameters']['index'])!=(m,n,idx):raise RuntimeError('Worker provenance mismatch')
        db.execute('INSERT OR REPLACE INTO attempts VALUES (?,?,?,?,?,?)',(m,int(n),idx,int(ok),reason,json.dumps(r,allow_nan=False)))
        pending+=1;done[m,n].add(idx)
        if ok:good[m,n].add(idx)
        if pending>=200 or time.monotonic()-last_commit[0]>10:
            db.commit();pending=0;last_commit[0]=time.monotonic();publish()
    publish();why=None
    try:
        while not stop['signal']:
            accepting=run_seconds is None or time.monotonic()-start<run_seconds
            while accepting and len(queue)<2*workers and claim():pass
            if pool is None:
                if not queue:break
                t=queue.popleft();record(t,worker(t));continue
            while queue and len(inflight)<2*workers:
                t=queue.popleft();inflight[pool.submit(worker,t)]=t
            if not inflight:break
            fin,_=wait(inflight,timeout=30,return_when=FIRST_COMPLETED)
            for f in fin:record(inflight.pop(f),f.result())
        if run_seconds is not None and time.monotonic()-start>=run_seconds:why='requested run duration reached'
    except (KeyboardInterrupt,BrokenProcessPool) as err:
        why='interrupted (%s)'%type(err).__name__
    finally:
        if stop['signal']:why='stopped by signal %d'%stop['signal']
        if pool:
            for f in inflight:f.cancel()           # (shutdown(cancel_futures=...) needs Python 3.9)
            pool.shutdown(wait=True)                # wait=False trips a Python 3.8 bug at exit
        db.commit();publish();db.close()
    return why


# ------------------------------------------------------------------ merge ----
def merge(output,write=True):
    """First `target` accepted inputs per case in proposal-index order, inside the contiguous
    range of completed indices of all workers together."""
    out=Path(output);spec=json.loads((out/'spec.json').read_text());target=spec['target_per_case']
    done,good,where,dbs=union(output,spec);cases=[];chosen={}
    for c in CASES:
        P=prefix(done[c]);acc=sorted(i for i in good[c] if i<P);chosen[c]=acc[:target]
        cases.append(dict(model=c[0],nonideal=c[1],completed_prefix=P,attempts_completed=len(done[c]),
                          accepted_in_prefix=len(acc),selected_count=min(len(acc),target),target=target,complete=len(acc)>=target))
    info=dict(sampler=SAMPLER_VERSION,engine_version=spec['engine_version'],target_per_case=target,workers=len(dbs),
              required_rows=target*len(CASES),selected_rows=sum(c['selected_count'] for c in cases),
              complete=all(c['complete'] for c in cases),test_dataset=spec['test_dataset'],cases=cases)
    if not write:
        for db in dbs:db.close()
        return info,None
    test=spec['test_dataset']
    name='fig_2_inputs'+('_test' if test else '')+('' if info['complete'] else '_partial')+'.csv';dest=out/name
    def rows():
        for c in CASES:
            for i in chosen[c]:
                raw=where[c][i].execute('SELECT result FROM attempts WHERE model=? AND nonideal=? AND sample_index=?',(c[0],int(c[1]),i)).fetchone()[0]
                r=json.loads(raw);row=flatten(r)
                row.update(lambda1=r['lambda_H'],test_dataset=test,sampler_version=SAMPLER_VERSION,
                           selection_rule='first_accepted_by_proposal_index',target_per_case=target)
                yield row
    fields=['point_id','model','nonideal','cls','lambda1','Delta1','B1','sigma_ps','sigma'];seen=set(fields)
    for row in rows():
        for k in row:
            if k not in seen:fields.append(k);seen.add(k)
    tmp=dest.with_suffix('.tmp');n=0
    with tmp.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for row in rows():w.writerow(row);n+=1
    if n!=info['selected_rows']:raise RuntimeError('CSV export count mismatch')
    tmp.replace(dest);info.update(csv_file=name,csv_rows=n,spec=spec)
    save_json(out/'fig_2_generation.json',info)
    for db in dbs:db.close()
    return info,dest


# ------------------------------------------------------------------- main ----
def main():
    import socket
    p=parser(__doc__)
    p.add_argument('--name',help='name of this worker process (default: host name + job id / process id)')
    p.add_argument('--margin',type=float,default=3.,help='extra accepted inputs claimed per case, in units of sqrt(target)')
    p.add_argument('--run-seconds',type=float,help='stop claiming new blocks after this many seconds')
    p.add_argument('--test-target',type=int,help='small verification run only; writes a distinctly named test CSV')
    p.add_argument('--init',action='store_true',help='create/check the run directory, free blocks of dead workers, exit')
    p.add_argument('--helper',action='store_true',help='join a run as an extra worker (other nodes); never merges')
    p.add_argument('--finish',action='store_true',help='main job: sample, then wait for the helpers (redoing the blocks of dead ones) and write the CSV')
    p.add_argument('--merge',action='store_true',help='only merge all workers and write the CSV')
    p.add_argument('--status',action='store_true',help='print the merged progress and exit')
    a=p.parse_args();target=TARGET if a.test_target is None else a.test_target
    if target<1 or a.workers<1:p.error('invalid target or workers')
    if a.test_target is None and a.preset!='production':p.error('Smoke settings require an explicit --test-target')
    if a.status or a.merge:
        info,dest=merge(a.output,write=a.merge);print(json.dumps(info,indent=2))
        if dest:print(dest)
        if a.merge and not info['complete']:raise SystemExit('INCOMPLETE: resubmit; the workers continue where they stopped.')
        return
    spec=dict(sampler_version=SAMPLER_VERSION,engine_version=VERSION,seed=a.seed,config=asdict(config(a)),
              target_per_case=target,test_dataset=a.test_target is not None,block=BLOCK)
    spec=init_run(a.output,spec)
    STALE=float(os.environ.get('FIG2_STALE_SECONDS',900))     # a worker without heartbeat for 15 min is gone
    if a.init:
        print('freed %d unfinished blocks of stopped workers'%release_stale(a.output,spec,STALE));return
    job=os.environ.get('PBS_JOBID','').split('.')[0] or str(os.getpid())
    name=a.name or socket.gethostname().split('.')[0]+'-'+job
    if a.helper:
        why=generate(spec,a.output,name,a.workers,a.run_seconds,a.margin)
        print('helper %s finished%s'%(name,'' if why is None else ' ('+why+')'));return
    if a.finish:
        start=time.monotonic();limit=(a.run_seconds or 1e9)+1800  # 30 min after the claiming stops
        while True:
            left=None if a.run_seconds is None else max(0.,a.run_seconds-(time.monotonic()-start))
            why=generate(spec,a.output,name,a.workers,left,a.margin)
            info,_=merge(a.output,write=False)
            if info['complete'] or why and why.startswith('stopped') or time.monotonic()-start>limit:break
            freed=release_stale(a.output,spec,STALE)          # helpers that died: redo their blocks
            if not freed:
                print('waiting for the helpers: %s'%', '.join('%s/%s %d/%d'%(c['model'],'non' if c['nonideal'] else 'id',
                      c['accepted_in_prefix'],c['target']) for c in info['cases']),flush=True)
                time.sleep(60)
    else:
        release_stale(a.output,spec)                           # desktop: nobody else is running
        why=generate(spec,a.output,name,a.workers,a.run_seconds,a.margin)
    info,dest=merge(a.output);print(dest)
    if not info['complete']:raise SystemExit('INCOMPLETE: '+(why or 'resume the same command')+'. No complete production CSV was written.')
    print('COMPLETE:',info['selected_rows'],'rows;',target,'per model per mixing case')

if __name__=='__main__':main()
