//============================================================
// StackingAction.cc
// Kill optical photons to speed up simulation (we track all
// charged secondaries and gammas already via SteppingAction)
//============================================================
#include "StackingAction.hh"
#include "RunAction.hh"
#include "G4Track.hh"
#include "G4ParticleDefinition.hh"
#include "G4OpticalPhoton.hh"

StackingAction::StackingAction(RunAction* run)
  : G4UserStackingAction(), fRunAction(run)
{}

StackingAction::~StackingAction() {}

G4ClassificationOfNewTrack
StackingAction::ClassifyNewTrack(const G4Track* track)
{
    // Kill optical photons – we don't simulate light transport
    if (track->GetDefinition() == G4OpticalPhoton::OpticalPhotonDefinition())
        return fKill;

    return fUrgent;
}
