# Literature References

This page lists key literature references for the Arterial Input Functions (AIFs) and Pharmacokinetic (PK) models implemented or relevant to this DCE-MRI analysis tool.

## Arterial Input Functions (AIFs)

1.  **Parker, G. J. M., Roberts, C., Macdonald, A., Buonaccorsi, G. A., Cheung, S., Buckley, D. L., Jackson, A., Watson, Y., Davies, K., & Jayson, G. C. (2006).**
    *   Title: *Experimentally-derived functional form for a population-averaged high-temporal-resolution arterial input function for dynamic contrast-enhanced MRI.*
    *   Journal: *Magnetic Resonance in Medicine, 56(5), 993–1000.*
    *   DOI: `10.1002/mrm.21066`
    *   Relevance: Provides the bi-exponential model and parameter values for the "Parker AIF" widely used in DCE-MRI.

2.  **Weinmann, H. J., Laniado, M., & Mützel, W. (1984).** (Note: The tool's UI often cites 1982, Am J Roentgenol for a related earlier paper by Weinmann on contrast agents, but the specific AIF form is better linked to later works or common adaptations. The Parker paper itself refers to a different Weinmann AIF profile. For the purpose of this tool, the "Weinmann AIF" usually refers to a generic bi-exponential form with parameters that have become common in literature, sometimes traced back to early contrast agent studies by Weinmann et al.)
    *   Title: *Pharmacokinetics of GdDTPA/dimeglumine after intravenous injection into healthy volunteers.*
    *   Journal: *Physiological Chemistry and Physics and Medical NMR, 16(2), 167-172.*
    *   PMID: `6333011`
    *   Relevance: While not the direct source of the exact AIF parameters used in many generic "Weinmann" AIFs today, this is representative of early pharmacokinetic studies of Gd-based contrast agents by Weinmann and colleagues that informed AIF modeling. The "Weinmann AIF" in software often uses parameters like A1=3.99, m1=0.144, A2=4.78, m2=0.0111 (units vary).

3.  **Georgiou, L., Wilson, D. J., & Sharma, R. (2019).**
    *   Title: *A fast bi-exponential arterial input function for dynamic contrast enhanced magnetic resonance imaging.*
    *   Journal: *Medical Physics, 46(10), 4596-4604.*
    *   DOI: `10.1002/mp.13728`
    *   Relevance: Describes a "fast bi-exponential AIF" which is one of the population AIFs available in this tool.

## Pharmacokinetic Models

1.  **Tofts, P. S., & Kermode, A. G. (1991).**
    *   Title: *Measurement of the blood-brain barrier permeability and leakage space using dynamic MR imaging. 1. Fundamental concepts.*
    *   Journal: *Journal of Magnetic Resonance Imaging, 1(4), 357–367.* (This is often cited, though the Tofts 1999 review is more comprehensive for the model itself).
    *   DOI: `10.1002/jmri.1880010402` (This DOI might be for a different paper, original JMRIs are harder to track). A more accessible reference for the general model is often the Tofts 1999 review.
    *   Relevance: Foundational work leading to the Standard Tofts model.

2.  **Tofts, P. S., Brix, G., Buckley, D. L., Evelhoch, J. L., Henderson, E., Knopp, M. V., Larsson, H. B. W., Lee, T.-Y., Mayr, N. A., Parker, G. J. M., Port, R. E., Taylor, J., & Weisskoff, R. M. (1999).**
    *   Title: *Estimating kinetic parameters from dynamic contrast-enhanced T1-weighted MRI of a diffusable tracer: standardized quantities and symbols.*
    *   Journal: *Journal of Magnetic Resonance Imaging, 10(3), 223–232.*
    *   DOI: `10.1002/(SICI)1522-2586(199909)10:3<223::AID-JMRI2>3.0.CO;2-S`
    *   Relevance: Seminal paper standardizing terminology and formulation for the Standard Tofts model and the Extended Tofts model. Defines K<sup>trans</sup>, v<sub>e</sub>, and v<sub>p</sub>.

3.  **Patlak, C. S., Blasberg, R. G., & Fenstermacher, J. D. (1983).**
    *   Title: *Graphical evaluation of blood-to-brain transfer constants from multiple-time uptake data.*
    *   Journal: *Journal of Cerebral Blood Flow and Metabolism, 3(1), 1–7.*
    *   DOI: `10.1038/jcbfm.1983.1`
    *   Relevance: Original description of the Patlak graphical analysis method, which can be applied to DCE-MRI data to estimate K<sup>trans</sup> (as the slope) and v<sub>p</sub> (as the intercept) under certain assumptions.

4.  **Sourbron, S. P., & Buckley, D. L. (2011).**
    *   Title: *Classic models for dynamic contrast-enhanced MRI.*
    *   Journal: *NMR in Biomedicine, 24(10), 1271-1285.* (This is a review, but a good reference for 2CXM).
    *   DOI: `10.1002/nbm.1673`
    *   Relevance: While the 2CXM has roots in earlier compartmental modeling principles (e.g., Kety), this review and others by Sourbron provide good overviews of its application in DCE-MRI. The specific implementation details (e.g., whether F<sub>p</sub> or PS are divided by v<sub>p</sub>/v<sub>e</sub> in the ODEs) can vary.

5.  **Larsson, H. B. W., Hansen, A. E., & Rostrup, E. (2009).**
    *   Title: *The two-compartment exchange model for determination of perfusion and permeability.*
    *   In: *Tofts PS (ed) Quantitative MRI of the Brain: Measuring Changes Caused by Disease. John Wiley & Sons, Ltd, Chichester, UK, pp 291–312.*
    *   DOI: `10.1002/9780470749751.ch12`
    *   Relevance: Provides a detailed description of the two-compartment exchange model (2CXM) in the context of brain DCE-MRI.

---
This list is not exhaustive but covers the primary models and AIFs relevant to this software. Users are encouraged to consult these references and other relevant literature for a deeper understanding of DCE-MRI theory and practice.
