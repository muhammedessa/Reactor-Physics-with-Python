HA1_11.ipynb
This assignment is a great way to understand the natural decay processes of uranium isotopes and how they affect uranium enrichment over geological time scales. The provided code should help you visualize and interpret these changes effectively.

What You Need to Do:
Understand the Derivation: Ensure you understand how the equation for uranium enrichment is derived, considering the decay of U-235 and U-238.

Run the Code: Execute the provided code to see the results and the plot.

Interpret the Results: Analyze the enrichment values at the specified times and understand how enrichment has changed over billions of years.

Optional Enhancements: If you want to explore further, you could modify the code to include additional time points or compare the enrichment values with other natural phenomena.



HA1-2.ipynb

This exercise provided valuable insights into the relationship between nuclear structure and stability. The nuclide chart revealed that nuclides with intermediate atomic numbers (e.g., iron-56) tend to have the highest binding energy per nucleon, making them the most stable. This aligns with the well-known trend in nuclear physics where heavier and lighter nuclides are less stable due to the balance between nuclear forces and Coulomb repulsion.

By completing this assignment, we gained practical experience in handling scientific data, performing physics calculations, and creating visualizations to interpret complex phenomena. These skills are essential for further studies and research in physics, particularly in fields like nuclear physics and computational modeling.



HA1-3.ipynb

The assignment provides a practical approach to handling and analyzing nuclear fuel inventory data, which is a routine task for nuclear reactor analysts. By completing this assignment, the student gains experience in:

File Handling and Data Extraction: Learning how to read and parse data from ASCII files, which is a common task in data analysis.

Data Visualization: Developing skills in plotting data to visualize trends and relationships, which is crucial for interpreting complex datasets.

Unit Conversion and Calculations: Understanding how to convert atomic concentrations to activity concentrations, which involves knowledge of nuclear physics and decay laws.

Critical Thinking and Analysis: Drawing meaningful conclusions from the data, which is essential for making informed decisions in nuclear reactor analysis.

The conclusions drawn from the analysis could include observations about how the concentration of Cs-137 changes with burnup and cooling time, the impact of different cooling time intervals on isotope concentrations, and the overall behavior of isotopes in spent nuclear fuel. These insights are valuable for understanding the long-term behavior of nuclear fuel and for making decisions related to nuclear waste management and reactor safety.



HA2-1.ipynb

Idea
The assignment appears to focus on nuclear physics concepts, specifically the neutron cycle and the four-factor formula used in nuclear reactor calculations. It involves visualizing the neutron lifecycle and understanding the factors influencing the infinite multiplication factor, including:

Fast fission factor (
𝜂
η)
Resonance escape probability (
𝑝
p)
Fast-to-thermal neutron utilization factor (
𝑓
f)
Thermal utilization factor (
𝜖
ϵ).
Purpose
Conceptual Understanding: To help the student grasp the components of the neutron cycle and the significance of each factor in nuclear reactor physics.
Practical Skill Development:
Enhance coding skills by modifying and completing a visualization using Python's matplotlib.
Calculate key parameters for a nuclear reactor's infinite multiplication factor.
Analytical Thinking: To interpret scientific data and translate theoretical concepts into numerical estimates and visual representations.
Conclusion
The assignment likely concludes with:

Modified Visualization: A corrected figure illustrating the neutron cycle with properly labeled factors (
𝑓
,
𝑝
,
𝜂
,
𝜖
f,p,η,ϵ) and numerical values replacing placeholders.
Calculated Factors: Estimates for each factor of the four-factor formula based on the given scenario.
Infinite Multiplication Factor (
𝑘
∞
k 
∞
​
 ): A calculated value derived from the formula 
𝑘
∞
=
𝜂
⋅
𝑝
⋅
𝑓
⋅
𝜖
k 
∞
​
 =η⋅p⋅f⋅ϵ, showing whether the reactor is subcritical, critical, or supercritical.


 HA2-2.ipynb



Analyzed
python
Always show details

Copy code
# Load and analyze the contents of the second uploaded Jupyter Notebook file to understand its context.
# Path to the new uploaded file
file_path_2 = "/mnt/data/HA2-2.ipynb"

# Load the Jupyter Notebook
with open(file_path_2, 'r') as file:
    notebook_content_2 = read(file, as_version=4)

# Extract the notebook cells for analysis
notebook_cells_2 = notebook_content_2['cells']

# Extracting text content from markdown and code cells to understand the idea and purpose
notebook_summary_2 = {
    "markdown": [],
    "code": []
}

for cell in notebook_cells_2:
    if cell['cell_type'] == 'markdown':
        notebook_summary_2["markdown"].append(cell['source'])
    elif cell['cell_type'] == 'code':
        notebook_summary_2["code"].append(cell['source'])

notebook_summary_2
Result
{'markdown': ['# Home assigment 2\n\nPlease give your name below:',
  '## Exercise 2\n\nNow you will put together the pieces you learnt in previous labs in order to write a continuous energy Monte Carlo particle transport simulation of a Uranium sphere! Your goal is to estimate the k-effective and to try to find the critical radius of a U-235 sphere. \n\nYour assumptions:\n\n- The geometry is a sphere\n- The sphere is made of pure U-235.\n- You only care about the following reactions: capture, fission and elastic scattering \n- All scattering is isotropic in the CM frame.\n- Neutrons emerge isotropically from fission.\n- You received the pointwise cross sections and the energy dependent nubar data in the /data folder.\n- You can neglect that in a fission event the number of generated neutrons is stochastic, and assume that always nubar neutrons are created.\n- For the prompt neutron energies you can sample the Watt-distribution (see lecture notes, or Datalab 4)\n- You do not need to track time (thus all neutrons can be considered to be prompt)\n- Initially launch neutrons from the center of the sphere, then store fission sites, and later sample the new fission sites from this "bank".\n\nYour tasks:\n\n1. Plot the the cross section data and the nubar data.\n2. Complete the support functions given below and the function `run()` in order to estimate the k-eigenvalue of a sphere with a continous energy 3D Monte Carlo particle transport simulation. The support functions are the ones which you saw in Datalab 5b (for example the direction transformations, elastic scattering etc.). Some of these functions you will need to update (eg. for the reaction type sampler, include fission). You can include other input parameters and set default values if you feel needed. For each neutron generation estimate the k-eigenvalue based on the initial number of neutrons and the new neutrons after the generation (as we did in Datalab 5a).\n3. Modify the return list in order to return and plot other data\n    - Plot the k-eigenvalue estimate vs the generation number\n    - Plot how the estimated mean k-eigenvalue converges. (use such figures to argue about reasonable values for `NGEN`, `NPG`, `NSKIP`). \n4. Investigate how the k-eigenvalue depends on the radius of the sphere. Visualize this with matplotlib.\n5. Find the critical radius. You can do this either with manual trial and error, or use an optimization method.\n\n\nHints: in this exercise you have to merge your knowledge from datalab 5a (ie. batchwise estimation of k-effective) and from datalab 5b (ie. tracking neutrons). If you are not sure about the validity of your results you can compare your findings with the values of critical radii from [Wikipedia](https://en.wikipedia.org/wiki/Critical_mass). Try to have similar order of magnitude results.\n\nTo be fair, in a real MC criticality calculation, the initial number of neutrons per cycle also fluctuates, and the k-eigenvalue is calculated with some power iteration. In that case some care needs to be taken to renormalize the number of events to be placed in the bank, in order to have more or less the same amount of starting neutrons in each batch, otherwise sub and supercritical systems would be problematic to be simulated (here p200-225 gives some details on that: https://mcnp.lanl.gov/pdf_files/la-ur-16-29043.pdf). You don\'t need to worry about these. We are satisfied with a simpler approach. Rather you will initiate the same amount of neutrons in each cycle, regardless how many were produced before, and we place every fission site into the bank, and sample the locations from that. We also do not require an initial guess for the k-eigenvalue (as you can see in the link for the power iteration based method, an initial guess is needed). \n\nIn the first few cycles when we launch neutrons only from the center, we will probably underestimate leakage, so the estimates of $k$ will be biased. Therefore NSKIP just means that the first NSKIP number of cycle estimates of the k-effective should not be taken into account when calculating the mean of the k-effective, since the spatial distribution of the fission source is still biased by our original source location, and not spread yet throughout the geometry. Actually for this simple geometry NSKIP plays a less important role, so if you are not certain about what it is, feel free to ignore it. \n\nTry not to overcomplicate the exercise. The function `run()` with docstrings and comments can be written in less than 80 lines of code. Below we collected all the supporting functions from Datalab 4 and 5, which you will need to use. Some of them you need to update or finish first. We also loaded the nuclear data.\n\nAlso, ideally the computation should not be too slow. Test first with small NGEN and NPG values (eg. 100 for both). This should already provide decent accuracy. I tested that on an older laptop this many batches and particles can be run within a minute without any vectorization. If you experience that your computation is much longer, there might be a mistake.',
  '',
  'Calculate Macroscopic Cross-Sections',
  '![image.png](attachment:image.png)',
  'Step 4: Implement the run() Function\nThe run() function performs the Monte Carlo simulation for a given sphere radius (\nR\nR), number of generations (\nN\nG\nE\nN\nNGEN), neutrons per generation (\nN\nP\nG\nNPG), and inactive generations (\nN\nS\nK\nI\nP\nNSKIP).',
  ' Run the Simulation and Analyze Results\nRun the simulation for different sphere radii to find the critical radius where \nk\neff\n≈\n1\nk \neff\n\u200b\n ≈1.',
  'Find the Critical Radius\nIterate over different radii to find the critical radius where \nk\neff\n≈\n1\nk \neff\n\u200b\n ≈1.'],
 'code': ["name=''",
  'import numpy as np\nimport random\nimport matplotlib.pyplot as plt\n\nEs,xss=np.loadtxt(\'data/u235el.dat\',skiprows=2).transpose()\nEc,xsc=np.loadtxt(\'data/u235cap.dat\',skiprows=2).transpose()\nEf,xsf=np.loadtxt(\'data/u235fiss.dat\',skiprows=2).transpose()\n\nEnu,nubar=np.loadtxt(\'data/u235nubar.dat\',skiprows=2).transpose()\n\n\ndensity = 19.1 #g/cm3\nA = 235\n\n#TODO : get the macroscopic cross section\n\n\n##### SUPPORT functions\ndef distanceToCollision(SigT,N=1):\n    x=np.random.uniform(0,1,N)\n    return -np.log(x)/SigT\n\ndef reactionType(SigS,SigC,SigT):\n    #TODO: include the fission cross section!\n    x=np.random.uniform(0,1)\n    if x < SigS/SigT:\n        return \'scatter\'\n    else:\n        return \'capture\'\n\ndef elasticScatter(E):\n    muC=np.random.uniform(-1,1)\n    thetaC=np.arccos(muC)\n    E=(((1+alpha)+(1-alpha)*muC)/2)*E\n    thetaL=np.arctan2(np.sin(thetaC),((1/A)+muC))\n    muL=np.cos(thetaL)\n    return E, muL\n\ndef randomDir():\n    mu=np.random.uniform(-1,1)\n    theta=np.arccos(mu)\n    phi=np.random.uniform(0,2*np.pi)\n\n    u=np.sin(theta)*np.cos(phi)\n    v=np.sin(theta)*np.sin(phi)\n    w=np.cos(theta)\n    return np.array([u,v,w])\n\ndef transformDir(u,v,w,mu):\n    """\n    transform coordinates according to openMC documentation.\n    \n    Parameters\n    ----------\n    u : float\n        Old x-direction\n    v : float\n        Old y-direction\n    w : float\n        Old z-direction\n    mu : float\n        Lab cosine of scattering angle\n    """\n    phi=np.random.uniform(0,2*np.pi)\n    un=mu*u+(np.sqrt(1-mu**2)*(u*w*np.cos(phi)-v*np.sin(phi)))/(np.sqrt(1-w**2))\n    vn=mu*v+(np.sqrt(1-mu**2)*(v*w*np.cos(phi)+u*np.sin(phi)))/(np.sqrt(1-w**2))\n    wn=mu*w-np.sqrt(1-mu**2)*np.sqrt(1-w**2)*np.cos(phi)\n    return np.array([un,vn,wn])\n\ndef watt(x): \n    """\n    Function to return the Watt distribution\n\n    Parameters\n    ----------\n    x : float\n        Energy in MeV\n    """\n    C1 = 0.453\n    C2 = 0.965\n    C3 = 2.29\n    return C1*np.exp(-x/C2)*np.sinh(np.sqrt(C3*x))\n\ndef wattrnd(N):\n    """\n    Function to return energies sampled from the Watt-distribution.\n\n    Parameters\n    ----------\n    N : int\n        Number of samples needed\n    """\n    #TODO: you have to complete this function. You can use the rejection method\n    # which we used in Datalab4. Just now the dots we plotted as "accepted" you \n    # can keep and return.\n    # NOTE: take care of the energy units! Make sure, that the energy unit you use\n    # here matches the energy unit used in your simulation!\n    return 0\n\n\n',
  'def run(R,NGEN,NPG,NSKIP):\n    """Function to perform a criticality calculation in a U-235 sphere.\n    \n    Parameters\n    ----------\n    R : float\n        Radius of the sphere\n    NGEN : int\n        Number of neutron generations\n    NPG : int\n        Number of neutrons per generation\n    NSKIP : int\n        Number of inactive generations which will not be taken into account for estimating the k-eigenvalue\n    \n    Returns\n    -------\n    keff : float\n        The estimated mean k-eigenvalue of the system\n    kstd : float\n        The standard deviation of the estimated k-eigenvalue\n    """\n    return keff, kstd',
  "import numpy as np\nimport matplotlib.pyplot as plt\n\n# Load cross-section and nubar data\nEs, xss = np.loadtxt('data/u235el.dat', skiprows=2).transpose()  # Elastic scattering\nEc, xsc = np.loadtxt('data/u235cap.dat', skiprows=2).transpose()  # Capture\nEf, xsf = np.loadtxt('data/u235fiss.dat', skiprows=2).transpose()  # Fission\nEnu, nubar = np.loadtxt('data/u235nubar.dat', skiprows=2).transpose()  # Nubar\n\n# Plot cross-sections\nplt.figure(figsize=(10, 6))\nplt.loglog(Es, xss, label='Elastic Scattering')\nplt.loglog(Ec, xsc, label='Capture')\nplt.loglog(Ef, xsf, label='Fission')\nplt.xlabel('Energy (eV)')\nplt.ylabel('Cross-Section (b)')\nplt.title('U-235 Cross-Sections')\nplt.legend()\nplt.grid(True)\nplt.show()\n\n# Plot nubar\nplt.figure(figsize=(10, 6))\nplt.semilogx(Enu, nubar, label='Nubar')\nplt.xlabel('Energy (eV)')\nplt.ylabel('Neutrons per Fission')\nplt.title('U-235 Nubar')\nplt.legend()\nplt.grid(True)\nplt.show()",
  "density = 19.1  # g/cm³\nA = 235  # Atomic mass of U-235\nNA = 6.022e23  # Avogadro's number\n\n# Convert microscopic cross-sections to macroscopic cross-sections\nSigS = (density * NA / A) * xss  # Elastic scattering\nSigC = (density * NA / A) * xsc  # Capture\nSigF = (density * NA / A) * xsf  # Fission\nSigT = SigS + SigC + SigF  # Total cross-section",
  'def distanceToCollision(SigT, N=1):\n    """Calculate the distance to the next collision."""\n    x = np.random.uniform(0, 1, N)\n    return -np.log(x) / SigT\n\ndef reactionType(SigS, SigC, SigF, SigT):\n    """Determine the type of reaction."""\n    x = np.random.uniform(0, 1)\n    if x < SigS / SigT:\n        return \'scatter\'\n    elif x < (SigS + SigC) / SigT:\n        return \'capture\'\n    else:\n        return \'fission\'\n\ndef elasticScatter(E):\n    """Simulate elastic scattering."""\n    muC = np.random.uniform(-1, 1)\n    thetaC = np.arccos(muC)\n    E = (((1 + alpha) + (1 - alpha) * muC) / 2) * E\n    thetaL = np.arctan2(np.sin(thetaC), ((1 / A) + muC))\n    muL = np.cos(thetaL)\n    return E, muL\n\ndef randomDir():\n    """Generate a random direction."""\n    mu = np.random.uniform(-1, 1)\n    theta = np.arccos(mu)\n    phi = np.random.uniform(0, 2 * np.pi)\n    u = np.sin(theta) * np.cos(phi)\n    v = np.sin(theta) * np.sin(phi)\n    w = np.cos(theta)\n    return np.array([u, v, w])\n\ndef transformDir(u, v, w, mu):\n    """Transform the direction vector after scattering."""\n    phi = np.random.uniform(0, 2 * np.pi)\n    un = mu * u + (np.sqrt(1 - mu**2) * (u * w * np.cos(phi) - v * np.sin(phi))) / np.sqrt(1 - w**2)\n    vn = mu * v + (np.sqrt(1 - mu**2) * (v * w * np.cos(phi) + u * np.sin(phi))) / np.sqrt(1 - w**2)\n    wn = mu * w - np.sqrt(1 - mu**2) * np.sqrt(1 - w**2) * np.cos(phi)\n    return np.array([un, vn, wn])\n\ndef watt(x):\n    """Watt distribution for fission neutron energies."""\n    C1 = 0.453\n    C2 = 0.965\n    C3 = 2.29\n    return C1 * np.exp(-x / C2) * np.sinh(np.sqrt(C3 * x))\n\ndef wattrnd(N):\n    """Sample energies from the Watt distribution using rejection sampling."""\n    energies = []\n    while len(energies) < N:\n        x = np.random.uniform(0, 10)  # Sample energy in MeV\n        y = np.random.uniform(0, 0.5)  # Sample uniform value\n        if y < watt(x):\n            energies.append(x)\n    return np.array(energies)',
  'def run(R, NGEN, NPG, NSKIP):\n    """Perform a criticality calculation in a U-235 sphere."""\n    k_eff = []\n    fission_sites = []\n\n    for gen in range(NGEN):\n        neutrons = []\n        if gen < NSKIP:\n            # Start neutrons from the center\n            for _ in range(NPG):\n                neutrons.append({\'pos\': np.array([0, 0, 0]), \'dir\': randomDir(), \'E\': 2.0})  # 2 MeV initial energy\n        else:\n            # Sample fission sites\n            for _ in range(NPG):\n                site = random.choice(fission_sites)\n                neutrons.append({\'pos\': site[\'pos\'], \'dir\': randomDir(), \'E\': wattrnd(1)[0]})\n\n        # Track neutrons\n        new_fission_sites = []\n        for neutron in neutrons:\n            while True:\n                # Distance to collision\n                d = distanceToCollision(SigT)\n                # Update position\n                neutron[\'pos\'] += d * neutron[\'dir\']\n                # Check if neutron escapes\n                if np.linalg.norm(neutron[\'pos\']) > R:\n                    break\n                # Determine reaction type\n                reaction = reactionType(SigS, SigC, SigF, SigT)\n                if reaction == \'scatter\':\n                    neutron[\'E\'], mu = elasticScatter(neutron[\'E\'])\n                    neutron[\'dir\'] = transformDir(*neutron[\'dir\'], mu)\n                elif reaction == \'capture\':\n                    break\n                elif reaction == \'fission\':\n                    new_fission_sites.append({\'pos\': neutron[\'pos\']})\n                    break\n\n        # Update fission sites\n        fission_sites = new_fission_sites\n        # Estimate k_eff\n        if gen >= NSKIP:\n            k_eff.append(len(fission_sites) / NPG)\n\n    # Calculate mean and standard deviation of k_eff\n    keff = np.mean(k_eff)\n    kstd = np.std(k_eff)\n    return keff, kstd',
  'from scipy.interpolate import interp1d\n\n# Interpolate cross-sections Interpolate Cross-Sections\n#We need to interpolate the cross-sections to get the macroscopic cross-section at any given energy. We can use np.interp for this.\nSigS_interp = interp1d(Es, SigS, kind=\'linear\', fill_value="extrapolate")\nSigC_interp = interp1d(Ec, SigC, kind=\'linear\', fill_value="extrapolate")\nSigF_interp = interp1d(Ef, SigF, kind=\'linear\', fill_value="extrapolate")\nSigT_interp = interp1d(Es, SigT, kind=\'linear\', fill_value="extrapolate")\n\n#The distanceToCollision function should now take the neutron\'s energy as input and return a single distance based on the interpolated cross-section.\ndef distanceToCollision(SigT_interp, E):\n    """Calculate the distance to the next collision for a neutron with energy E."""\n    SigT = SigT_interp(E)  # Get macroscopic cross-section at energy E\n    x = np.random.uniform(0, 1)\n    return -np.log(x) / SigT\n\n#Similarly, the reactionType function should take the neutron\'s energy as input and use the interpolated cross-sections.\ndef reactionType(SigS_interp, SigC_interp, SigF_interp, SigT_interp, E):\n    """Determine the type of reaction for a neutron with energy E."""\n    SigS = SigS_interp(E)\n    SigC = SigC_interp(E)\n    SigF = SigF_interp(E)\n    SigT = SigT_interp(E)\n    x = np.random.uniform(0, 1)\n    if x < SigS / SigT:\n        return \'scatter\'\n    elif x < (SigS + SigC) / SigT:\n        return \'capture\'\n    else:\n        return \'fission\'\n    \n\n#Now, update the run() function to use the interpolated cross-sections and the updated distanceToCollision and reactionType functions.\ndef run(R, NGEN, NPG, NSKIP):\n    """Perform a criticality calculation in a U-235 sphere."""\n    k_eff = []\n    fission_sites = []\n\n    for gen in range(NGEN):\n        neutrons = []\n        if gen < NSKIP:\n            # Start neutrons from the center\n            for _ in range(NPG):\n                neutrons.append({\'pos\': np.array([0.0, 0.0, 0.0]), \'dir\': randomDir(), \'E\': 2.0})  # 2 MeV initial energy\n        else:\n            # Sample fission sites\n            for _ in range(NPG):\n                site = random.choice(fission_sites)\n                neutrons.append({\'pos\': site[\'pos\'], \'dir\': randomDir(), \'E\': wattrnd(1)[0]})\n\n        # Track neutrons\n        new_fission_sites = []\n        for neutron in neutrons:\n            while True:\n                # Distance to collision\n                d = distanceToCollision(SigT_interp, neutron[\'E\'])\n                # Update position\n                neutron[\'pos\'] += d * neutron[\'dir\']\n                # Check if neutron escapes\n                if np.linalg.norm(neutron[\'pos\']) > R:\n                    break\n                # Determine reaction type\n                reaction = reactionType(SigS_interp, SigC_interp, SigF_interp, SigT_interp, neutron[\'E\'])\n                if reaction == \'scatter\':\n                    neutron[\'E\'], mu = elasticScatter(neutron[\'E\'])\n                    neutron[\'dir\'] = transformDir(*neutron[\'dir\'], mu)\n                elif reaction == \'capture\':\n                    break\n                elif reaction == \'fission\':\n                    new_fission_sites.append({\'pos\': neutron[\'pos\']})\n                    break\n\n        # Update fission sites\n        fission_sites = new_fission_sites\n        # Estimate k_eff\n        if gen >= NSKIP:\n            k_eff.append(len(fission_sites) / NPG)\n\n    # Calculate mean and standard deviation of k_eff\n    keff = np.mean(k_eff)\n    kstd = np.std(k_eff)\n    return keff, kstd\n\n\n\n# Constants\ndensity = 19.1  # g/cm³\nA = 235  # Atomic mass of U-235\nNA = 6.022e23  # Avogadro\'s number\nalpha = ((A - 1) / (A + 1)) ** 2  # Elastic scattering parameter\n\nkeff, kstd = run(R, NGEN, NPG, NSKIP)\nprint(f"k_eff = {keff:.4f} ± {kstd:.4f}")\n\n\ndef elasticScatter(E):\n    """Simulate elastic scattering."""\n    muC = np.random.uniform(-1, 1)  # Cosine of scattering angle in CM frame\n    thetaC = np.arccos(muC)  # Scattering angle in CM frame\n    # Energy after scattering\n    E = (((1 + alpha) + (1 - alpha) * muC) / 2) * E\n    # Lab scattering angle\n    thetaL = np.arctan2(np.sin(thetaC), ((1 / A) + muC))\n    muL = np.cos(thetaL)  # Cosine of scattering angle in Lab frame\n    return E, muL\n\n#Now, test the updated simulation with the same parameters.\n# Test the simulation \nR = 10  # Radius in cm\nNGEN = 100  # Number of generations\nNPG = 100  # Neutrons per generation\nNSKIP = 10  # Inactive generations\n\nkeff, kstd = run(R, NGEN, NPG, NSKIP)\nprint(f"k_eff = {keff:.4f} ± {kstd:.4f}")\n\n\n\n\n\nradii = np.linspace(5, 15, 10)  # Test radii from 5 cm to 15 cm\nk_effs = []\n\nfor R in radii:\n    keff, _ = run(R, NGEN, NPG, NSKIP)\n    k_effs.append(keff)\n    print(f"R = {R:.2f} cm, k_eff = {keff:.4f}")\n\n# Plot k_eff vs radius\nplt.plot(radii, k_effs, \'o-\')\nplt.axhline(1, color=\'red\', linestyle=\'--\', label=\'Critical (k_eff = 1)\')\nplt.xlabel(\'Radius (cm)\')\nplt.ylabel(\'k_eff\')\nplt.title(\'k_eff vs Sphere Radius\')\nplt.legend()\nplt.grid(True)\nplt.show()',
  'radii = np.linspace(5, 15, 10)  # Test radii from 5 cm to 15 cm\nk_effs = []\n\nfor R in radii:\n    keff, _ = run(R, NGEN, NPG, NSKIP)\n    k_effs.append(keff)\n    print(f"R = {R:.2f} cm, k_eff = {keff:.4f}")\n\n# Plot k_eff vs radius\nplt.plot(radii, k_effs, \'o-\')\nplt.axhline(1, color=\'red\', linestyle=\'--\', label=\'Critical (k_eff = 1)\')\nplt.xlabel(\'Radius (cm)\')\nplt.ylabel(\'k_eff\')\nplt.title(\'k_eff vs Sphere Radius\')\nplt.legend()\nplt.grid(True)\nplt.show()',
  '']}
Idea
This assignment is centered around simulating neutron transport and determining the criticality of a uranium sphere using a Monte Carlo approach. The focus is on estimating the effective neutron multiplication factor (
𝑘
𝑒
𝑓
𝑓
k 
eff
​
 ) and finding the critical radius of a U-235 sphere.

Purpose
Practical Application of Physics and Computational Techniques:

Simulate neutron interactions such as elastic scattering, fission, and capture.
Use cross-section and energy-dependent data for U-235 to model neutron behavior.
Criticality Analysis:

Estimate 
𝑘
𝑒
𝑓
𝑓
k 
eff
​
  to understand whether the sphere is subcritical, critical, or supercritical.
Identify the sphere's critical radius (where 
𝑘
𝑒
𝑓
𝑓
≈
1
k 
eff
​
 ≈1) by analyzing its geometry and material properties.
Skill Development:

Apply Monte Carlo simulations to real-world nuclear engineering problems.
Implement and test scientific algorithms for stochastic processes.
Visualization:

Plot cross-section and 
𝑘
𝑒
𝑓
𝑓
k 
eff
​
  data.
Visualize convergence of 
𝑘
𝑒
𝑓
𝑓
k 
eff
​
  and its dependency on sphere radius.
Conclusion
The assignment likely concludes with:

Simulation Outputs:

A plot of 
𝑘
𝑒
𝑓
𝑓
k 
eff
​
  versus sphere radius showing how criticality is achieved.
Convergence of 
𝑘
𝑒
𝑓
𝑓
k 
eff
​
  over successive neutron generations.
Critical Radius Determination:

Identification of the radius where 
𝑘
𝑒
𝑓
𝑓
≈
1
k 
eff
​
 ≈1, ensuring the system is in criticality.
Comparison of results with literature values (e.g., Wikipedia critical mass data).
Insights:

How 
𝑘
𝑒
𝑓
𝑓
k 
eff
​
  depends on geometric and material properties.
The importance of accurate cross-section data and simulation parameters.








