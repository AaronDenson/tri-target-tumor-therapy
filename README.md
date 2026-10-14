# TRI-TARGET-TUMOR THERAPY (T4)

Glioblastoma multiforme is the deadliest human cancer and amongst the most belligerent malignancies known to mankind, with a nearly guaranteed fatality exhibiting a survival rate of 5-7% (Ostrom et al., 2015). 

Furthermore, GBM accounts for new yearly cases in the U.S. alone (Ostrom et al., 2015). despite decades of research, current GBM therapies lack the precision required to eradicate tumor cells without damaging healthy brain Tissue and traverse through the Blood-Brain Barrier (BBB). On a common basis, chemotherapy treatments like Temozolomide (TMZ) imposes severe side effects like myelosuppression, liver damage, and long-term cognitive decline due to the low precision.

<img width="439" height="53" alt="image" src="https://github.com/user-attachments/assets/c4796f8a-dd84-4d23-b606-76961a6ef642" />


To solve this, we propose T4 (Tri-Target Tumor Therapy)! T-4 aligns closely with its **4 T's**:

1.) **T**ARGET: T4 was computationally modeled to traverse the BBB with literature-based parameters and docking.

2.) **T**EST: T4 evaluates 3 independent GBM prevalent biomarkers of EGFRvIII, MMP-2, and miR-210.		

3.) **T**RIGGER: Activation thresholds for triple-lock AND-logic gate were optimized with quantum-inspired optimization via QAOA simulated on classical hardware.

4.) **T**REAT: Upon gate activation, T4 releases STAT3 siRNA. Performance is measured by a therapeutic index.

<img width="320" height="309" alt="image" src="https://github.com/user-attachments/assets/0dac24f5-1cb4-4c32-849a-9193c947680b" />

**_Our Objective_:
**

**Primary Endpoints:** Quantitatively compare therapeutic index (TPR/FPR) for single-, dual-, and triple-lock gates on simulated GBM vs healthy-brain cell populations and show that the triple-lock structure of the nanogate achieves the highest therapeutic index.

**Secondary Endpoints:** **G**enerate gate activation across layered tumor models (core, invasive rim, healthy-like tissue) to confirm triple-lock confinement to the GBM core. **B**enchmark multi-cycle treatment and biomarker drift to assess gate durability and whether triple-lock gating delays tumor escape relative to single- and dual-lock designs. **M**otivate the triple-lock and STAT3 siRNA design with molecular docking results.

Essentially, the mathematics within the problem line up with this:
<img width="893" height="118" alt="image" src="https://github.com/user-attachments/assets/980edcc7-a8f0-4ee2-bcb1-efe84ff64841" />

To start, the Engineering Process works through four distinct steps: Design, Lock Test, Simulate, and Optimize:

<img width="579" height="425" alt="image" src="https://github.com/user-attachments/assets/2e7ad97f-5ee8-4352-afd9-64c84174c923" />

**How T4 Works:
**

T4 integrates blood-brain barrier (BBB) transport modeling, molecular targeting, and triple-biomarker logic into a unified computational framework for glioblastoma therapy. First, literature-informed transport kinetics model drug movement across the BBB. 

<img width="662" height="484" alt="image" src="https://github.com/user-attachments/assets/2aa8f0d2-e7db-4095-943a-afd600e22b53" />


Next, the system evaluates three GBM-associated biomarkers—EGFRvIII, MMP-2, and miR-210—through a triple-lock AND gate, requiring all three activation conditions to be satisfied. Threshold optimization and Monte Carlo simulations then evaluate the gate's selectivity across simulated tumor and healthy-brain populations. Upon activation, the proposed system is designed to release STAT3-targeting siRNA, with performance assessed using a therapeutic index comparing tumor activation against unintended healthy-tissue activation.

<img width="661" height="481" alt="image" src="https://github.com/user-attachments/assets/281b7901-6200-40c0-bb3b-f0d27fdc67f7" />


The central hypothesis: By requiring three independent biomarker conditions, T4 aims to improve tumor-selective activation while minimizing off-target effects on healthy brain tissue.

<img width="1348" height="377" alt="image" src="https://github.com/user-attachments/assets/03753a8c-1fa8-4f0b-b788-ffb59063c7e0" />
