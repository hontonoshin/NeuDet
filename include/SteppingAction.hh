//============================================================
// SteppingAction.hh
//============================================================
#ifndef SteppingAction_h
#define SteppingAction_h 1

#include "G4UserSteppingAction.hh"
#include "globals.hh"

class EventAction;
class DetectorConstruction;

class SteppingAction : public G4UserSteppingAction
{
public:
    SteppingAction(DetectorConstruction* det, EventAction* evt);
    ~SteppingAction() override;

    void UserSteppingAction(const G4Step* step) override;

private:
    DetectorConstruction* fDetector;
    EventAction*          fEventAction;
};

#endif
