"""Create reproducible, entirely fictional week-five LMS export data (stdlib only)."""
import csv
import hashlib
import json
import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 903
COHORTS = [('2023_S1','2023-02-06'),('2023_S2','2023-09-04'),('2024_S1','2024-02-05'),('2024_S2','2024-09-02'),('2025_S1','2025-02-03'),('2025_S2','2025-09-01')]
FIELDS = ['tenant_id','student_key','course_id','cohort_id','snapshot_at','course_start_at','course_end_at','enrolled_at','enrolment_status_at_snapshot','activity_coverage_start_at','activity_complete_through_at','active_days_14d','active_days_previous_14d','interactions_14d','interactions_previous_14d','resource_views_14d','forum_posts_14d','last_activity_at','assignments_due_to_date','assignments_submitted_to_date','late_submissions_to_date','mean_score_pct_to_date','scores_available_count','source_system','schema_version']
LABEL_FIELDS = ['tenant_id','student_key','course_id','cohort_id','snapshot_at','final_outcome','withdrawn_at','outcome_observed_at']
def stamp(dt): return dt.strftime('%Y-%m-%dT%H:%M:%SZ')
def save(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
def generate():
    rng=random.Random(SEED); rows=[]; labels=[]; counts={}
    for ci,(cohort,day) in enumerate(COHORTS):
        start=datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
        cutoff=start+timedelta(days=35); end=start+timedelta(days=112)
        positives=0
        for i in range(600):
            engagement=rng.betavariate(3,2)
            if rng.random()<0.18: engagement*=0.3
            prior=min(14,max(0,round(engagement*14+rng.gauss(0,2))))
            active=min(14,max(0,round(prior+rng.gauss(-0.5,3))))
            clicks=active*rng.randint(3,20); prev=prior*rng.randint(3,20)
            idle=rng.randint(0,14-active) if active else (rng.randint(14,28-prior) if prior else rng.randint(28,35))
            last='' if idle==35 else stamp(cutoff-timedelta(days=idle,hours=rng.randint(1,23)))
            # A no-activity learner can still have historical activity before this 28-day window.
            due=2+(i%3); submitted=sum(rng.random() < (0.20+0.77*engagement) for _ in range(due))
            late=sum(rng.random() < (0.48-0.35*engagement) for _ in range(submitted))
            marked=sum(rng.random()<0.88 for _ in range(submitted))
            score=round(min(100,max(0,43+40*engagement+rng.gauss(0,13))),1) if marked else ''
            enrolled=start+timedelta(days=rng.randint(-7,3))
            if last and datetime.fromisoformat(last.replace('Z','+00:00')) < enrolled: last=''
            views=rng.randint(0,clicks); posts=rng.randint(0,min(8,clicks-views))
            row=dict(zip(FIELDS, ['demo_university',f'SYN_{ci*600+i+1:06d}',f'COURSE_{i%6+1:02d}',cohort,stamp(cutoff),stamp(start),stamp(end),stamp(enrolled),'active',stamp(start),stamp(cutoff),active,prior,clicks,prev,views,posts,last,due,submitted,late,score,marked,'synthetic_moodle','1.0']))
            # Bernoulli outcomes are noisy simulations, not deterministic rules or real evidence.
            z=-2.8 + 0.10*idle + 0.62*(due-submitted)+0.32*late + 0.07*(prior-active) + rng.gauss(0,0.75)
            if score!='': z+=0.025*(60-score)
            drop=int(rng.random()<1/(1+math.exp(-z))); positives+=drop
            outcome='withdrawn' if drop else ('failed' if rng.random()<0.12 else 'completed')
            label={k:row[k] for k in LABEL_FIELDS[:5]}
            label.update(final_outcome=outcome,withdrawn_at=stamp(cutoff+timedelta(days=rng.randint(1,76))) if drop else '',outcome_observed_at=stamp(end+timedelta(days=14)))
            rows.append(row); labels.append(label)
        counts[cohort]={'rows':600,'withdrawals':positives}
    save(ROOT/'data/raw/lms_snapshots.csv',FIELDS,rows)
    save(ROOT/'data/raw/student_outcomes.csv',LABEL_FIELDS,labels)
    # Separate serving input has the same feature schema and no outcome columns.
    save(ROOT/'data/examples/scoring_input.csv',FIELDS,rows[-12:])
    invalid=[dict(rows[-1])]; invalid[0]['active_days_14d']=19
    save(ROOT/'data/fixtures/invalid_snapshot.csv',FIELDS,invalid)
    drift=[]
    for source in rows[-600:]:
        r=dict(source); r.update(active_days_14d=0,interactions_14d=0,resource_views_14d=0,forum_posts_14d=0,last_activity_at=stamp(datetime.fromisoformat(r['snapshot_at'].replace('Z','+00:00'))-timedelta(days=14 if r['active_days_previous_14d'] else 30, hours=1)))
        drift.append(r)
    save(ROOT/'data/fixtures/drifted_scoring_input.csv',FIELDS,drift)
    manifest={'synthetic':True,'generator':'scripts/generate_synthetic_data.py','seed':SEED,'schema_version':'1.0','prediction':'withdraw after day 35 and by course end, among active enrolments','row_unit':'one student-course enrolment at day 35','cohorts':counts,'split':{'train':[c[0] for c in COHORTS[:4]],'validation':[COHORTS[4][0]],'test':[COHORTS[5][0]]},'limitations':['Fictional relationships and outcomes; no real-world accuracy or calibration claims.','Scoring and drift examples reuse held-out fictional records and must never enter training.','Fixed day-35 snapshot only; weekly scoring requires a revised training design.'],'files':{}}
    for rel in ['data/raw/lms_snapshots.csv','data/raw/student_outcomes.csv','data/examples/scoring_input.csv','data/fixtures/invalid_snapshot.csv','data/fixtures/drifted_scoring_input.csv']:
        manifest['files'][rel]={'sha256':hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()}
    (ROOT/'data/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'rows':len(rows),'cohorts':counts,'withdrawal_rate':round(sum(x['final_outcome']=='withdrawn' for x in labels)/len(labels),4)},indent=2))
if __name__=='__main__': generate()
