//============================================================
// DetectorConstruction.hh
// Li6ZnS:Ag neutron scintillator detector
//============================================================
#ifndef DetectorConstruction_h
#define DetectorConstruction_h 1

#include "G4VUserDetectorConstruction.hh"
#include "G4Material.hh"
#include "G4LogicalVolume.hh"
#include "G4VPhysicalVolume.hh"
#include "globals.hh"

class DetectorConstruction : public G4VUserDetectorConstruction
{
public:
    DetectorConstruction();
    ~DetectorConstruction() override;

    G4VPhysicalVolume* Construct() override;
    void ConstructSDandField() override;

    G4double GetDetThickness() const { return fDetThickness; }
    G4double GetDetSizeXY()    const { return fDetSizeXY; }
    G4LogicalVolume* GetScoringVolume() const { return fScoringVolume; }

private:
    void DefineMaterials();

    // World dimensions
    G4double fWorldSizeXY  = 30.0;   // cm
    G4double fWorldSizeZ   = 50.0;   // cm

    // Detector slab (Li6ZnS:Ag)
    G4double fDetSizeXY    = 20.0;   // cm
    G4double fDetThickness = 0.5;    // cm  (5 mm – typical Li6ZnS sheet)

    // Thin scorer planes (vacuum, 10 µm)
    G4double fScorerThick  = 0.001;  // cm

    G4Material* fScintMat;
    G4Material* fWorldMat;

    G4LogicalVolume* fScoringVolume;   // the scintillator itself
    G4LogicalVolume* fLogicPreScore;
    G4LogicalVolume* fLogicPostScore;
};

#endif
