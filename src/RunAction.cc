//============================================================
// RunAction.cc
//
// Output files:
//   pre_energy.csv        – incident neutron energies [MeV]
//   post_energy.csv       – transmitted neutron energies [MeV]
//   edep.csv              – total energy deposition per event [keV]
//   secondaries.csv       – secondary particle data
//   processes.csv         – interaction process tally
//   summary.txt           – run statistics: efficiency, attenuation, etc.
//   path_edep.csv         – per-step edep in scintillator [keV]
//============================================================
#include "RunAction.hh"
#include "PrimaryGeneratorAction.hh"
#include "G4Run.hh"
#include "G4SystemOfUnits.hh"
#include <iomanip>
#include <cmath>

RunAction::RunAction(PrimaryGeneratorAction* pga)
  : G4UserRunAction(),
    fPGA(pga),
    fNTotal(0),
    fNAbsorbed(0),
    fNInteracted(0),
    fNTransmitted(0)
{}

RunAction::~RunAction() {}

void RunAction::BeginOfRunAction(const G4Run*)
{
    fNTotal = fNAbsorbed = fNInteracted = fNTransmitted = 0;
    fPreEnergyList.clear();
    fPostEnergyList.clear();
    fEdepList.clear();
    fPathEdepList.clear();
    fSecondaries.clear();
    fProcessCount.clear();

    // Open output files
    fPreEnergyFile.open("pre_energy.csv");
    fPreEnergyFile << "energy_MeV\n";

    fPostEnergyFile.open("post_energy.csv");
    fPostEnergyFile << "energy_MeV\n";

    fEdepFile.open("edep.csv");
    fEdepFile << "edep_keV\n";

    fSecondaryFile.open("secondaries.csv");
    fSecondaryFile << "particle,ek_MeV,px,py,pz,rigidity_Tm\n";

    fProcessFile.open("processes.csv");
    // written at end of run

    fSummaryFile.open("summary.txt");
}

void RunAction::EndOfRunAction(const G4Run* run)
{
    G4long nEvents = run->GetNumberOfEvent();

    // Write path edep
    std::ofstream pathFile("path_edep.csv");
    pathFile << "edep_keV\n";
    for (double v : fPathEdepList) pathFile << v << "\n";
    pathFile.close();

    // Write process counts
    fProcessFile << "process,count,fraction\n";
    for (auto& kv : fProcessCount) {
        double frac = (fNInteracted > 0) ? (double)kv.second/fNInteracted : 0;
        fProcessFile << kv.first << "," << kv.second << ","
                     << std::fixed << std::setprecision(4) << frac << "\n";
    }
    fProcessFile.close();

    //----------------------------------------------------------
    // Detection efficiency
    // ε_det = N_absorbed / N_incident
    //----------------------------------------------------------
    double effAbs    = (fNTotal > 0) ? (double)fNAbsorbed   / fNTotal : 0.;
    double effInter  = (fNTotal > 0) ? (double)fNInteracted / fNTotal : 0.;
    double effTrans  = (fNTotal > 0) ? (double)fNTransmitted/ fNTotal : 0.;
    double effStat   = (fNAbsorbed > 0) ? std::sqrt((double)fNAbsorbed)/fNTotal : 0.;

    // Attenuation (macroscopic removal estimate)
    double attenuation = 1.0 - effTrans;

    //----------------------------------------------------------
    // Summary file
    //----------------------------------------------------------
    fSummaryFile << "============================================\n";
    fSummaryFile << " Li6ZnS:Ag Neutron Detector – Run Summary\n";
    fSummaryFile << "============================================\n";
    fSummaryFile << " Total primaries generated  : " << fNTotal        << "\n";
    fSummaryFile << " Neutrons entering detector  : " << fPreEnergyList.size()  << "\n";
    fSummaryFile << " Transmitted neutrons        : " << fNTransmitted  << "\n";
    fSummaryFile << " Neutrons interacted         : " << fNInteracted   << "\n";
    fSummaryFile << " Neutrons absorbed (killed)  : " << fNAbsorbed     << "\n";
    fSummaryFile << "--------------------------------------------\n";
    fSummaryFile << " Intrinsic efficiency ε_abs  : "
                 << std::fixed << std::setprecision(4) << effAbs*100. << " %\n";
    fSummaryFile << " Interaction probability     : "
                 << effInter*100. << " %\n";
    fSummaryFile << " Transmission fraction       : "
                 << effTrans*100. << " %\n";
    fSummaryFile << " Attenuation fraction        : "
                 << attenuation*100. << " %\n";
    fSummaryFile << " Statistical uncertainty (ε) : "
                 << effStat*100. << " %\n";
    fSummaryFile << "--------------------------------------------\n";

    // Mean/RMS edep
    if (!fEdepList.empty()) {
        double sum=0., sum2=0.;
        for (double v: fEdepList){ sum+=v; sum2+=v*v; }
        double mean = sum/fEdepList.size();
        double rms  = std::sqrt(sum2/fEdepList.size() - mean*mean);
        fSummaryFile << " Mean edep per event (keV)   : "
                     << std::fixed << std::setprecision(2) << mean << "\n";
        fSummaryFile << " RMS edep (keV)              : " << rms << "\n";
    }

    // Process breakdown
    fSummaryFile << "\n Process breakdown (interacted):\n";
    for (auto& kv : fProcessCount) {
        double frac = (fNInteracted > 0) ? 100.*kv.second/fNInteracted : 0;
        fSummaryFile << "   " << std::setw(35) << std::left << kv.first
                     << " : " << kv.second << "  (" << frac << "%)\n";
    }
    fSummaryFile << "============================================\n";

    // Print to terminal too
    G4cout << "\n====== Run Summary ======\n"
           << " Events           : " << fNTotal << "\n"
           << " Efficiency (abs) : " << effAbs*100. << " %\n"
           << " Interaction prob : " << effInter*100. << " %\n"
           << " Transmission     : " << effTrans*100. << " %\n"
           << "=========================\n";

    fPreEnergyFile.close();
    fPostEnergyFile.close();
    fEdepFile.close();
    fSecondaryFile.close();
    fSummaryFile.close();
}

void RunAction::FillPreEnergy(G4double e)  {
    fPreEnergyList.push_back(e);
    fPreEnergyFile << std::setprecision(8) << e << "\n";
}
void RunAction::FillPostEnergy(G4double e) {
    fPostEnergyList.push_back(e);
    fPostEnergyFile << std::setprecision(8) << e << "\n";
}
void RunAction::FillEdep(G4double e) {
    fEdepList.push_back(e);
    fEdepFile << std::setprecision(6) << e << "\n";
}
void RunAction::FillNeutronPathEdep(G4double e) {
    fPathEdepList.push_back(e);
}
void RunAction::FillSecondary(const std::string& name, G4double ek,
                              G4double px, G4double py, G4double pz)
{
    fSecondaries.push_back({name, ek, px, py, pz});

    // Magnetic rigidity Bρ [T·m] for charged particles
    // Bρ = p / (Z·e·c) = p[MeV/c] / (Z × 299.792 MeV/(T·m))
    double mass_MeV = 0.;
    int    charge_z = 0;
    if      (name == "triton")  { mass_MeV = 2808.921; charge_z = 1; }
    else if (name == "alpha")   { mass_MeV = 3727.379; charge_z = 2; }
    else if (name == "proton")  { mass_MeV = 938.272;  charge_z = 1; }
    else if (name == "e-")      { mass_MeV = 0.511;    charge_z = 1; }
    else if (name == "e+")      { mass_MeV = 0.511;    charge_z = 1; }

    double rigidity = -1.;
    if (charge_z > 0 && mass_MeV > 0.) {
        double E_total = ek + mass_MeV; // MeV
        double p = std::sqrt(E_total*E_total - mass_MeV*mass_MeV); // MeV/c
        rigidity = p / (charge_z * 299.792); // T·m
    }

    fSecondaryFile << name << ","
                   << std::setprecision(8) << ek << ","
                   << px << "," << py << "," << pz << ","
                   << rigidity << "\n";
}
void RunAction::RecordProcess(const std::string& proc) {
    fProcessCount[proc]++;
}
void RunAction::IncrementTotalEvents()  { fNTotal++; }
void RunAction::IncrementAbsorbed()     { fNAbsorbed++; }
void RunAction::IncrementInteracted()   { fNInteracted++; }
void RunAction::IncrementTransmitted()  { fNTransmitted++; }
