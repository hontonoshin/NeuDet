//============================================================
// SteppingAction.cc
//
// Scored quantities per step:
//   1. Pre-detector neutron energy (primary neutron entering PreScorer)
//   2. Post-detector neutron energy (primary neutron in PostScorer)
//   3. Energy deposition in scintillator
//   4. Secondary particle type, kinetic energy, momentum direction
//   5. Interaction process name
//   6. Whether neutron was absorbed (track killed in scintillator)
//   7. Per-step edep along neutron path (for dose map)
//
// Rigidity (magnetic rigidity Bρ) for charged secondaries:
//   Bρ [T·m] = p [MeV/c] / (Z × c [MeV/(T·m)])
//             = p[MeV/c] / (Z × 299.792)
//   Useful for discriminating triton vs alpha in Li6(n,t)4He
//============================================================
#include "SteppingAction.hh"
#include "EventAction.hh"
#include "DetectorConstruction.hh"

#include "G4Step.hh"
#include "G4Track.hh"
#include "G4ParticleDefinition.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include "G4VProcess.hh"
#include "G4LogicalVolume.hh"
#include "G4RunManager.hh"
#include "G4Triton.hh"
#include "G4Alpha.hh"
#include "G4Proton.hh"
#include "G4Gamma.hh"
// G4StrUtil not available in all Geant4 11.x builds; use std::string::find

#include <cmath>

SteppingAction::SteppingAction(DetectorConstruction* det, EventAction* evt)
  : G4UserSteppingAction(),
    fDetector(det),
    fEventAction(evt)
{}

SteppingAction::~SteppingAction() {}

void SteppingAction::UserSteppingAction(const G4Step* step)
{
    G4Track* track = step->GetTrack();
    G4ParticleDefinition* particle = track->GetDefinition();
    G4String particleName = particle->GetParticleName();

    G4LogicalVolume* preVolume =
        step->GetPreStepPoint()->GetTouchableHandle()->GetVolume()->GetLogicalVolume();
    G4LogicalVolume* postVolumeLV = nullptr;
    if (step->GetPostStepPoint()->GetTouchableHandle()->GetVolume())
        postVolumeLV = step->GetPostStepPoint()->GetTouchableHandle()->GetVolume()->GetLogicalVolume();

    G4String preVolName  = preVolume ? preVolume->GetName() : "";
    G4String postVolName = postVolumeLV ? postVolumeLV->GetName() : "OutOfWorld";

    G4LogicalVolume* scoringVol = fDetector->GetScoringVolume();
    bool inScintillator = (preVolume == scoringVol);

    //==========================================================
    // 1. Score primary neutron at pre-scorer plane
    //==========================================================
    if (particleName == "neutron" && track->GetParentID() == 0) {
        if (preVolName == "PreScorer") {
            G4double ekin = step->GetPreStepPoint()->GetKineticEnergy();
            fEventAction->AddPreNeutronEnergy(ekin / MeV);
        }

        //======================================================
        // 2. Score transmitted neutron at post-scorer plane
        //======================================================
        if (preVolName == "PostScorer") {
            G4double ekin = step->GetPreStepPoint()->GetKineticEnergy();
            fEventAction->AddPostNeutronEnergy(ekin / MeV);
        }

        //======================================================
        // 3. Check if neutron was absorbed inside scintillator
        //======================================================
        if (inScintillator) {
            const G4VProcess* proc = step->GetPostStepPoint()->GetProcessDefinedStep();
            if (proc) {
                G4String procName = proc->GetProcessName();
                if (track->GetTrackStatus() == fStopAndKill) {
                    fEventAction->SetNeutronAbsorbed(true);
                    fEventAction->SetInteractionProcess(procName);
                }
                auto procStd = std::string(procName);
                if (procStd.find("NeutronHP")  != std::string::npos ||
                    procStd.find("hadElastic") != std::string::npos ||
                    procStd.find("Inelastic")  != std::string::npos ||
                    procStd.find("nCapture")   != std::string::npos ||
                    procStd.find("nFission")   != std::string::npos)
                {
                    fEventAction->SetNeutronInteracted(true);
                    fEventAction->SetInteractionProcess(procName);
                }
            }

            // Per-step edep (from neutron recoils in elastic scatter)
            G4double edep = step->GetTotalEnergyDeposit();
            if (edep > 0.) fEventAction->AddNeutronStep(edep, 0,0,0,
                step->GetPreStepPoint()->GetKineticEnergy()/MeV);
        }
    }

    //==========================================================
    // 4. Energy deposition in scintillator (ALL particles)
    //==========================================================
    if (inScintillator) {
        G4double edep = step->GetTotalEnergyDeposit();
        if (edep > 0.) fEventAction->AddEdep(edep);
    }

    //==========================================================
    // 5. Score secondary particles created in scintillator
    //==========================================================
    // Look at newly created secondaries of this step
    if (inScintillator && track->GetParentID() >= 0) {
        const std::vector<const G4Track*>* secondaries =
            step->GetSecondaryInCurrentStep();
        if (secondaries) {
            for (const G4Track* sec : *secondaries) {
                G4String   secName = sec->GetDefinition()->GetParticleName();
                G4double   ek      = sec->GetKineticEnergy();
                G4ThreeVector mom  = sec->GetMomentumDirection();

                fEventAction->AddSecondary(secName, ek,
                    mom.x(), mom.y(), mom.z());
            }
        }
    }
}
