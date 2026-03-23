//============================================================
// StackingAction.hh
//============================================================
#ifndef StackingAction_h
#define StackingAction_h 1

#include "G4UserStackingAction.hh"
#include "globals.hh"

class RunAction;

class StackingAction : public G4UserStackingAction
{
public:
    StackingAction(RunAction* run);
    ~StackingAction() override;
    G4ClassificationOfNewTrack ClassifyNewTrack(const G4Track* track) override;
private:
    RunAction* fRunAction;
};

#endif
