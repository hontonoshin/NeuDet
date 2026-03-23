//============================================================
// ActionInitialization.hh
//============================================================
#ifndef ActionInitialization_h
#define ActionInitialization_h 1

#include "G4VUserActionInitialization.hh"

class DetectorConstruction;

class ActionInitialization : public G4VUserActionInitialization
{
public:
    ActionInitialization(DetectorConstruction* det);
    ~ActionInitialization() override;

    void BuildForMaster() const override;
    void Build() const override;

private:
    DetectorConstruction* fDetector;
};

#endif
