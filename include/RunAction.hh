//============================================================
// RunAction.hh
//============================================================
#ifndef RunAction_h
#define RunAction_h 1

#include "G4UserRunAction.hh"
#include "globals.hh"
#include <fstream>
#include <string>
#include <map>
#include <vector>

class PrimaryGeneratorAction;

class RunAction : public G4UserRunAction
{
public:
    RunAction(PrimaryGeneratorAction* pga);
    ~RunAction() override;

    void BeginOfRunAction(const G4Run* run) override;
    void EndOfRunAction(const G4Run* run)   override;

    // Accumulators filled by EventAction
    void FillPreEnergy(G4double e);
    void FillPostEnergy(G4double e);
    void FillEdep(G4double e);
    void FillSecondary(const std::string& name, G4double ek,
                       G4double px, G4double py, G4double pz);
    void IncrementTotalEvents();
    void IncrementAbsorbed();
    void IncrementInteracted();
    void IncrementTransmitted();
    void RecordProcess(const std::string& proc);
    void FillNeutronPathEdep(G4double edep);

private:
    PrimaryGeneratorAction* fPGA;

    // Counters
    G4long fNTotal;
    G4long fNAbsorbed;
    G4long fNInteracted;
    G4long fNTransmitted;

    // Histograms (stored as vectors for CSV output)
    std::vector<G4double> fPreEnergyList;
    std::vector<G4double> fPostEnergyList;
    std::vector<G4double> fEdepList;
    std::vector<G4double> fPathEdepList;

    struct SecEntry {
        std::string name;
        G4double    ek, px, py, pz;
    };
    std::vector<SecEntry> fSecondaries;

    std::map<std::string,G4long> fProcessCount;

    // Output file streams
    std::ofstream fPreEnergyFile;
    std::ofstream fPostEnergyFile;
    std::ofstream fEdepFile;
    std::ofstream fSecondaryFile;
    std::ofstream fSummaryFile;
    std::ofstream fProcessFile;
};

#endif
