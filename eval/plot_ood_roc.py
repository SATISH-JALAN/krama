"""
Renders eval/results/guardrail_calibration.json's marginal ROC curves as
eval/results/ood_roc.png, per ARCHITECTURE.md §8.2's explicit deliverable
("Commit eval/results/ood_roc.png"). Plots the two guardrails' individual
ROC shapes (diagnostic) and marks the joint operating point that actually
ships in thresholds.json, so the plot honestly shows both.
"""
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("eval/results/guardrail_calibration.json", encoding="utf-8") as f:
    report = json.load(f)

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# --- L1 safety marginal ROC ---
safety_roc = report["safetyMarginal"]["roc"]
fpr = [p["fpr"] for p in safety_roc]
tpr = [p["tpr"] for p in safety_roc]
axes[0].plot(fpr, tpr, color="#2e7d5b", linewidth=2)
axes[0].axvline(0.05, color="#999", linestyle="--", linewidth=1, label="5% FPR bound")
m = report["safetyMarginal"]
axes[0].scatter([m["fprInDomain"]], [m["tprInjectionsOnly"]], color="#c1440e", zorder=5,
                label=f"marginal op. point (tau={m['chosenThreshold']})")
axes[0].set_xlabel("False positive rate (in-domain flagged unsafe)")
axes[0].set_ylabel("True positive rate (injections caught)")
axes[0].set_title("L1 safety: exemplar-similarity ROC")
axes[0].legend(fontsize=8)
axes[0].set_xlim(-0.02, 1.02)
axes[0].set_ylim(-0.02, 1.02)

# --- L2 OOD marginal ROC (score-only sweep) ---
ood_roc = report["oodMarginal"]["rocScoreOnly"]
fpr2 = [p["fpr"] for p in ood_roc]
tpr2 = [p["tpr"] for p in ood_roc]
axes[1].plot(fpr2, tpr2, color="#2e7d5b", linewidth=2)
axes[1].axvline(0.05, color="#999", linestyle="--", linewidth=1, label="5% FPR bound")
om = report["oodMarginal"]
axes[1].scatter([om["fprInDomain"]], [om["tprOodAll"]], color="#c1440e", zorder=5,
                label=f"marginal op. point (tau1={om['chosenThresholds']['minTopScore']})")
jp = report["jointOperatingPoint"]
axes[1].scatter([jp["fprInDomain"]], [jp["tprOodAll"]], color="#1f6feb", marker="D", zorder=5,
                label=f"joint op. point (ships) FPR={jp['fprInDomain']:.1%}")
axes[1].set_xlabel("False positive rate (in-domain flagged off-topic)")
axes[1].set_ylabel("True positive rate (OOD queries caught)")
axes[1].set_title("L2 OOD: top-retrieval-score ROC")
axes[1].legend(fontsize=8)
axes[1].set_xlim(-0.02, 1.02)
axes[1].set_ylim(-0.02, 1.02)

fig.suptitle("KRAMA guardrail calibration -- PLAN.md E5.5 (500 in-domain / 199 hand-written OOD)")
fig.tight_layout()
fig.savefig("eval/results/ood_roc.png", dpi=150)
print("wrote eval/results/ood_roc.png")
