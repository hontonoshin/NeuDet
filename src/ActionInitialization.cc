//============================================================
// ActionInitialization.cc
//============================================================
#include "ActionInitialization.hh"
#include "PrimaryGeneratorAction.hh"
#include "RunAction.hh"
#include "EventAction.hh"
#include "SteppingAction.hh"
#include "StackingAction.hh"

ActionInitialization::ActionInitialization(DetectorConstruction* det)
  : G4VUserActionInitialization(), fDetector(det)
{}

ActionInitialization::~ActionInitialization() {}

void ActionInitialization::BuildForMaster() const
{
    // Master thread only needs RunAction
    auto* pga = new PrimaryGeneratorAction();
    SetUserAction(new RunAction(pga));
}

void ActionInitialization::Build() const
{
    auto* pga     = new PrimaryGeneratorAction();
    auto* runAct  = new RunAction(pga);
    auto* evtAct  = new EventAction(runAct);

    SetUserAction(pga);
    SetUserAction(runAct);
    SetUserAction(evtAct);
    SetUserAction(new SteppingAction(fDetector, evtAct));
    SetUserAction(new StackingAction(runAct));
}
