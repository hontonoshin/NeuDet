//============================================================
// DetectorConstruction.cc
//
// Geometry:
//   World (air, 30×30×50 cm)
//     |-- PreScorer  (vacuum, 10 µm slab)  z=-0.256 cm
//     |-- Scintillator (Li6ZnS:Ag, 20×20×0.5 cm) at z=0
//     |-- PostScorer (vacuum, 10 µm slab)  z=+0.256 cm
//
// Material: Li6ZnS:Ag
//   ZnS host with 6Li doping
//   Based on Eljen EJ-426 / Saint-Gobain B-10 literature:
//   Composition by mass fraction (approximate):
//     6Li  : 0.0275  (enriched 6Li)
//     Zn   : 0.4892
//     S    : 0.4833
//   Density: 3.18 g/cm3 (Knoll, Radiation Detection, 4th ed.)
//   Ag activator is trace (~0.01%) – not included in bulk
//============================================================
#include "DetectorConstruction.hh"
#include "G4NistManager.hh"
#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4SystemOfUnits.hh"
#include "G4SDManager.hh"
#include "G4MultiFunctionalDetector.hh"
#include "G4PSNofStep.hh"
#include "G4PSEnergyDeposit.hh"
#include "G4VisAttributes.hh"
#include "G4Isotope.hh"
#include "G4Element.hh"
#include "G4Material.hh"

DetectorConstruction::DetectorConstruction()
  : G4VUserDetectorConstruction(),
    fScoringVolume(nullptr),
    fLogicPreScore(nullptr),
    fLogicPostScore(nullptr),
    fScintMat(nullptr),
    fWorldMat(nullptr)
{}

DetectorConstruction::~DetectorConstruction() {}

void DetectorConstruction::DefineMaterials()
{
    G4NistManager* nist = G4NistManager::Instance();

    //--- World: dry air ---
    fWorldMat = nist->FindOrBuildMaterial("G4_AIR");

    //--- Li6ZnS:Ag scintillator ---
    // Isotopically enriched 6Li (95% enrichment, typical for detectors)
    G4Isotope* Li6 = new G4Isotope("Li6", 3, 6, 6.015*g/mole);
    G4Isotope* Li7 = new G4Isotope("Li7", 3, 7, 7.016*g/mole);

    G4Element* eLi = new G4Element("Enriched_Lithium","Li",2);
    eLi->AddIsotope(Li6, 0.95);   // 95% 6Li enrichment
    eLi->AddIsotope(Li7, 0.05);

    G4Element* eZn = nist->FindOrBuildElement("Zn");
    G4Element* eS  = nist->FindOrBuildElement("S");

    // Density from Knoll (2010) and manufacturer datasheets
    fScintMat = new G4Material("Li6ZnS_Ag", 3.18*g/cm3, 3);
    // Mass fractions from stoichiometry + Li doping level
    // ZnS: Zn 67.1%, S 32.9% by mass; Li doping ~5.5% by mass
    fScintMat->AddElement(eLi, 0.0275);
    fScintMat->AddElement(eZn, 0.4892);
    fScintMat->AddElement(eS,  0.4833);

    G4cout << "\n=== Material: Li6ZnS:Ag ===" << G4endl;
    G4cout << *fScintMat << G4endl;
}

G4VPhysicalVolume* DetectorConstruction::Construct()
{
    DefineMaterials();

    //================================================================
    // World
    //================================================================
    G4Box* solidWorld = new G4Box("World",
        0.5*fWorldSizeXY*cm, 0.5*fWorldSizeXY*cm, 0.5*fWorldSizeZ*cm);
    G4LogicalVolume* logicWorld = new G4LogicalVolume(solidWorld, fWorldMat, "World");
    G4VPhysicalVolume* physWorld = new G4PVPlacement(
        nullptr, G4ThreeVector(), logicWorld, "World", nullptr, false, 0, true);

    //================================================================
    // Pre-scorer plane (upstream face of detector)
    //================================================================
    G4Box* solidPre = new G4Box("PreScorer",
        0.5*fDetSizeXY*cm, 0.5*fDetSizeXY*cm, 0.5*fScorerThick*cm);
    fLogicPreScore = new G4LogicalVolume(solidPre, fWorldMat, "PreScorer");
    G4double preZ = -(0.5*fDetThickness + fScorerThick)*cm;
    new G4PVPlacement(nullptr, G4ThreeVector(0,0,preZ),
                      fLogicPreScore, "PreScorer", logicWorld, false, 0, true);

    //================================================================
    // Scintillator slab
    //================================================================
    G4Box* solidDet = new G4Box("Scintillator",
        0.5*fDetSizeXY*cm, 0.5*fDetSizeXY*cm, 0.5*fDetThickness*cm);
    fScoringVolume = new G4LogicalVolume(solidDet, fScintMat, "Scintillator");
    new G4PVPlacement(nullptr, G4ThreeVector(),
                      fScoringVolume, "Scintillator", logicWorld, false, 0, true);

    //================================================================
    // Post-scorer plane (downstream face)
    //================================================================
    G4Box* solidPost = new G4Box("PostScorer",
        0.5*fDetSizeXY*cm, 0.5*fDetSizeXY*cm, 0.5*fScorerThick*cm);
    fLogicPostScore = new G4LogicalVolume(solidPost, fWorldMat, "PostScorer");
    G4double postZ = +(0.5*fDetThickness + fScorerThick)*cm;
    new G4PVPlacement(nullptr, G4ThreeVector(0,0,postZ),
                      fLogicPostScore, "PostScorer", logicWorld, false, 0, true);

    return physWorld;
}

void DetectorConstruction::ConstructSDandField()
{
    // Sensitive detector registration is handled via SteppingAction
    // to give maximum flexibility for scoring all quantities
}
