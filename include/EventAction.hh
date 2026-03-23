//============================================================
// EventAction.hh
//============================================================
#ifndef EventAction_h
#define EventAction_h 1

#include "G4UserEventAction.hh"
#include "globals.hh"
#include <vector>
#include <string>

class RunAction;

class EventAction : public G4UserEventAction
{
public:
    EventAction(RunAction* runAction);
    ~EventAction() override;

    void BeginOfEventAction(const G4Event* event) override;
    void EndOfEventAction(const G4Event* event) override;

    // Called from SteppingAction
    void AddPreNeutronEnergy(G4double e)    { fPreNeutronEnergy = e;   fHasPreNeutron  = true; }
    void AddPostNeutronEnergy(G4double e)   { fPostNeutronEnergy = e;  fHasPostNeutron = true; }
    void AddEdep(G4double e)                { fEdepTotal += e; }
    void AddSecondary(const std::string& name, G4double ek, G4double px, G4double py, G4double pz);
    void SetNeutronAbsorbed(bool v)         { fNeutronAbsorbed = v; }
    void SetNeutronInteracted(bool v)       { fNeutronInteracted = v; }
    void SetInteractionProcess(const std::string& proc) { fInteractionProcess = proc; }
    void AddNeutronStep(G4double edep, G4double x, G4double y, G4double z, G4double ekin);

    G4double GetPreNeutronEnergy()  const { return fPreNeutronEnergy; }
    G4double GetPostNeutronEnergy() const { return fPostNeutronEnergy; }

private:
    RunAction* fRunAction;

    G4double fPreNeutronEnergy;
    G4double fPostNeutronEnergy;
    G4double fEdepTotal;
    bool     fHasPreNeutron;
    bool     fHasPostNeutron;
    bool     fNeutronAbsorbed;
    bool     fNeutronInteracted;
    std::string fInteractionProcess;

    struct SecondaryInfo {
        std::string name;
        G4double    ek;
        G4double    px, py, pz;
    };
    std::vector<SecondaryInfo> fSecondaries;

    struct NeutronStepInfo {
        G4double edep, x, y, z, ekin;
    };
    std::vector<NeutronStepInfo> fNeutronSteps;
};

#endif
