//============================================================
// EventAction.cc
//============================================================
#include "EventAction.hh"
#include "RunAction.hh"
#include "G4Event.hh"
#include "G4SystemOfUnits.hh"

EventAction::EventAction(RunAction* runAction)
  : G4UserEventAction(),
    fRunAction(runAction),
    fPreNeutronEnergy(0.),
    fPostNeutronEnergy(0.),
    fEdepTotal(0.),
    fHasPreNeutron(false),
    fHasPostNeutron(false),
    fNeutronAbsorbed(false),
    fNeutronInteracted(false)
{}

EventAction::~EventAction() {}

void EventAction::BeginOfEventAction(const G4Event*)
{
    fPreNeutronEnergy  = 0.;
    fPostNeutronEnergy = 0.;
    fEdepTotal         = 0.;
    fHasPreNeutron     = false;
    fHasPostNeutron    = false;
    fNeutronAbsorbed   = false;
    fNeutronInteracted = false;
    fInteractionProcess.clear();
    fSecondaries.clear();
    fNeutronSteps.clear();

    fRunAction->IncrementTotalEvents();
}

void EventAction::EndOfEventAction(const G4Event*)
{
    // Pre-detector flux
    if (fHasPreNeutron)
        fRunAction->FillPreEnergy(fPreNeutronEnergy);

    // Post-detector flux (transmitted neutrons)
    if (fHasPostNeutron) {
        fRunAction->FillPostEnergy(fPostNeutronEnergy);
        fRunAction->IncrementTransmitted();
    }

    // Energy deposition
    if (fEdepTotal > 0.)
        fRunAction->FillEdep(fEdepTotal / keV);

    // Secondaries
    for (auto& s : fSecondaries)
        fRunAction->FillSecondary(s.name, s.ek/MeV, s.px, s.py, s.pz);

    // Counters
    if (fNeutronAbsorbed)   fRunAction->IncrementAbsorbed();
    if (fNeutronInteracted) fRunAction->IncrementInteracted();
    if (!fInteractionProcess.empty())
        fRunAction->RecordProcess(fInteractionProcess);

    // Path-resolved edep
    for (auto& ns : fNeutronSteps)
        fRunAction->FillNeutronPathEdep(ns.edep / keV);
}

void EventAction::AddSecondary(const std::string& name, G4double ek,
                               G4double px, G4double py, G4double pz)
{
    fSecondaries.push_back({name, ek, px, py, pz});
}

void EventAction::AddNeutronStep(G4double edep, G4double x, G4double y,
                                 G4double z, G4double ekin)
{
    fNeutronSteps.push_back({edep, x, y, z, ekin});
}
