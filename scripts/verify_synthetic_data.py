"""Check delivered synthetic data, intentionally bad fixtures and regeneration hashes."""
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
KEY=['tenant_id','student_key','course_id','cohort_id','snapshot_at']
def read(rel):
    with (ROOT/rel).open(newline='') as f: return list(csv.DictReader(f))
def ts(s): return datetime.fromisoformat(s.replace('Z','+00:00'))
def key(r): return tuple(r[k] for k in KEY)
def validate(r):
    cutoff=ts(r['snapshot_at']); start=ts(r['course_start_at'])
    assert cutoff-start==timedelta(days=35)
    assert ts(r['enrolled_at'])<cutoff<ts(r['course_end_at'])
    assert r['enrolment_status_at_snapshot']=='active'
    assert ts(r['activity_coverage_start_at'])<=cutoff-timedelta(days=28)
    assert ts(r['activity_complete_through_at'])>=cutoff
    active=int(r['active_days_14d']); prior=int(r['active_days_previous_14d'])
    assert 0<=active<=14 and 0<=prior<=14
    clicks=int(r['interactions_14d']); previous=int(r['interactions_previous_14d'])
    assert clicks>=active and previous>=prior
    assert (clicks==0)==(active==0) and (previous==0)==(prior==0)
    assert 0<=int(r['resource_views_14d'])+int(r['forum_posts_14d'])<=clicks
    due=int(r['assignments_due_to_date']); submitted=int(r['assignments_submitted_to_date'])
    assert 0<=int(r['late_submissions_to_date'])<=submitted<=due
    marked=int(r['scores_available_count'])
    assert 0<=marked<=submitted
    assert (r['mean_score_pct_to_date']=='')==(marked==0)
    if marked: assert 0<=float(r['mean_score_pct_to_date'])<=100
    if r['last_activity_at']:
        last=ts(r['last_activity_at'])
        assert ts(r['enrolled_at'])<=last<cutoff
        assert (last>=cutoff-timedelta(days=14))==(active>0)
        if active==0:
            assert (last>=cutoff-timedelta(days=28))==(prior>0)
    else: assert active==prior==0

rows=read('data/raw/lms_snapshots.csv'); labels=read('data/raw/student_outcomes.csv')
assert len(rows)==len(labels)==3600
assert len({key(r) for r in rows})==len(rows)
assert {key(r) for r in rows}=={key(r) for r in labels}
by_key={key(r):r for r in rows}
for r in rows: validate(r)
for r in labels:
    source=by_key[key(r)]
    assert r['final_outcome'] in {'withdrawn','failed','completed'}
    assert ts(r['outcome_observed_at'])>=ts(source['course_end_at'])
    if r['final_outcome']=='withdrawn':
        assert ts(source['snapshot_at'])<ts(r['withdrawn_at'])<=ts(source['course_end_at'])
    else: assert r['withdrawn_at']==''
for r in read('data/examples/scoring_input.csv'): validate(r)
for r in read('data/fixtures/drifted_scoring_input.csv'): validate(r)
try: validate(read('data/fixtures/invalid_snapshot.csv')[0])
except AssertionError: pass
else: raise AssertionError('Invalid fixture unexpectedly passed')
manifest=json.loads((ROOT/'data/manifest.json').read_text())
for path,details in manifest['files'].items():
    assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==details['sha256']
counts=Counter(r['cohort_id'] for r in rows)
assert len(counts)==6 and set(counts.values())=={600}
assert not set(KEY)&{'final_outcome','withdrawn_at','outcome_observed_at'}
assert not {'final_outcome','withdrawn_at','outcome_observed_at','dropout_label'}&set(rows[0])
print(json.dumps({'status':'passed','snapshot_rows':len(rows),'outcome_rows':len(labels),'cohort_rows':dict(counts),'withdrawals':sum(r['final_outcome']=='withdrawn' for r in labels),'invalid_fixture':'rejected','drift_fixture':'schema-valid; drift detection planned','checksums':'matched'},indent=2))
