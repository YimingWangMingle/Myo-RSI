"""Supporting verification: source lineage, rejected branches, paired errors and costs."""
import json
import shutil
from collections import Counter
from pathlib import Path
import numpy as np

from .registry import ROOT,PROTOCOL,cohorts,completed,read,sha,table,write
from .report import rows


def supplement():
    rows_out=[]
    matching=[]
    blocked=[]
    sources=[]
    for s in read(PROTOCOL)["source_seeds"]:
        d=completed("diagnose",s)
        base=completed("screen",s)
        scr=completed("S2-screen",s)
        mon=read(d/"manifest.json")
        screening=read(base/"manifest.json")
        source={
            "source_seed":s,"checkpoint":mon["source_checkpoint"],"bank":mon["synergy_bank"],
            "official_dataset_sha256":sha("results/v0_1r/dataset_manifest.json"),
            "source_screen_run":base.name,"conditional_screen_run":scr.name,
            "body_sha256":{k:v["model"]["model_hash"] for k,v in screening["environment"].items()},
            "device":mon["environment"]["device"],"versions":mon["environment"]["versions"],
            "finger_order":"Native reach tip order from reward/command contract",
        }
        source["body_sha256"]["S2"]=mon["environment"]["model"]["model_hash"]
        sources.append(source)
        normal=rows(base/"T1.csv")
        for site,path in (("S1",base/"T3.csv"),("S2",scr/"S2_K16.csv")):
            weak=rows(path)
            for a,b in zip(normal,weak,strict=True):
                if (a["goal_hash"],a["reset_hash"])!=(b["goal_hash"],b["reset_hash"]):raise ValueError("Non-paired screening")
                rows_out.append(dict(seed=s,goal=a["goal"],site=site,normal=int(a["success"]),weak=int(b["success"]),
                       success_change=int(b["success"])-int(a["success"]),error_change_m=float(b["error"])-float(a["error"])))
        info=read(d/"diagnostic_report.json")
        bygoal={}
        for r in rows(d/"counterfactual_trials.csv"):
            bygoal.setdefault((int(r["goal"]),r["space"]),[]).append(r)
        for (goal,space),rr in bygoal.items():
            offset=0 if space=="K16" else 200
            with np.load(d/f"search_{goal}.npz") as z:
                m=z["match"][offset:offset+200]
                err=z["error"][-1,offset:offset+200]
            c=info["matches"][f"search_{goal}"]
            matching.append(dict(seed=s,goal=goal,space=space,prebranch_valid=int(m.sum()),
               registered_control_state_valid=bool(m[0]),repeat_control_state_valid=bool(m[1]),
               control_terminal_error=float(err[0]),accepted_after_control=sum(r["matched"]=="True" for r in rr),
               success_witnesses=sum(r["label"]=="success_witness" for r in rr),max_qpos_deviation=c["qpos"],
               max_qvel_deviation=c["qvel"],max_excitation_deviation=c["ctrl"],max_activation_deviation=c["act"]))
        if len(info["eligible_goals"])<2:
            blocked.append(dict(method="B3",seed=s,status="BLOCKED_BEFORE_TRAINING",diagnostic_run=d.name,
                diagnostic_transitions=sum(mon["budget"].values()),ppo_transitions=0,bc_updates=0,
                eligible_bc_goals=len(info["eligible_goals"]),required_distinct_goals=2,
                reason="No independent training and validation cases under registered matching and improvement rules"))
    write(ROOT/"source_manifest.json",sources)
    shutil.copyfile(PROTOCOL,ROOT/"protocol.json")
    table(ROOT/"screen_paired.csv",rows_out)
    table(ROOT/"replay_audit.csv",matching)
    if blocked:table(ROOT/"blocked_arms.csv",blocked)
    write(ROOT/"data_contract.json",dict(control_dt_s=.02,observation="115D actor obs before control, unnormalized",
        muscle_signals="native signed action; physical excitation ctrl; muscle activation act; raw recruitment and scaled post-ReLU coefficients",
        recorded_trajectories="failure_traces.npz includes 100 complete goals per seed; arrays (time,goal,channel); tip coordinates and all five errors; joint positions, velocities, warmstart, actuator length/velocity and reward terms",
        contacts="No contact-state trajectory was recorded; no contact-specific diagnosis is made.",
        tip_channels="five sites in the native Reach command order",
        search="Search arrays include replay match flags, candidate noise and complete post-branch error trajectories. corrections.npz contains only qualified fixed-bank training pairs and available full-muscle witness directions.",
        matching="Per-lane physical tolerances checked before branching; unchanged reference in each 200-lane block must also match and reproduce the original outcome and terminal error to 1 mm. Candidate counts are correlated within goal.",
        prior="Frozen v0.1r official teacher 100000 historical interactions; 18M source PPO transitions per seed. External teacher compute unavailable."))


if __name__=="__main__":
    supplement()
