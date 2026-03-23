//============================================================
// PhysicsList.cc
//
// Uses QGSP_BIC_HP which includes:
//   - G4HadronPhysicsQGSP_BIC_HP  (high-precision neutron transport
//     with ENDF/B-VIII cross-section data below 20 MeV)
//   - G4NeutronHPCapture, G4NeutronHPElastic, G4NeutronHPInelastic
//   - Thermal neutron scattering (S(alpha,beta)) via
//     G4NeutronHPThermalScattering
//   - EM physics (G4EmStandardPhysics_option4) for secondaries
//   - Radioactive decay (G4RadioactiveDecayPhysics)
//
// Key cross-section for Li6:
//   6Li(n,t)4He  Q = 4.78 MeV   σ_thermal ~ 940 barns
//   Products: triton (2.73 MeV) + alpha (2.05 MeV)
//============================================================
#include "PhysicsList.hh"

#include "G4VModularPhysicsList.hh"
#include "G4HadronPhysicsQGSP_BIC_HP.hh"
#include "G4HadronElasticPhysicsHP.hh"
#include "G4NeutronTrackingCut.hh"
#include "G4EmStandardPhysics_option4.hh"
#include "G4EmExtraPhysics.hh"
#include "G4DecayPhysics.hh"
#include "G4RadioactiveDecayPhysics.hh"
#include "G4StoppingPhysics.hh"
#include "G4IonPhysicsXS.hh"
#include "G4IonElasticPhysics.hh"
#include "G4ThermalNeutrons.hh"
#include "G4SystemOfUnits.hh"

PhysicsList::PhysicsList() : G4VModularPhysicsList()
{
    SetVerboseLevel(1);

    // EM physics with best precision for secondaries
    RegisterPhysics(new G4EmStandardPhysics_option4(verboseLevel));
    RegisterPhysics(new G4EmExtraPhysics(verboseLevel));

    // Decay
    RegisterPhysics(new G4DecayPhysics(verboseLevel));
    RegisterPhysics(new G4RadioactiveDecayPhysics(verboseLevel));

    // Hadron elastic with HP
    RegisterPhysics(new G4HadronElasticPhysicsHP(verboseLevel));

    // Hadron inelastic with HP (ENDF/B-VIII neutron data)
    RegisterPhysics(new G4HadronPhysicsQGSP_BIC_HP(verboseLevel));

    // Thermal neutron scattering (S(alpha,beta) for ZnS matrix)
    RegisterPhysics(new G4ThermalNeutrons(verboseLevel));

    // Stopping
    RegisterPhysics(new G4StoppingPhysics(verboseLevel));

    // Ion physics
    RegisterPhysics(new G4IonElasticPhysics(verboseLevel));
    RegisterPhysics(new G4IonPhysicsXS(verboseLevel));

    // Neutron tracking cut (kill neutrons below 1 meV – below thermal range)
    RegisterPhysics(new G4NeutronTrackingCut(verboseLevel));
}

PhysicsList::~PhysicsList() {}

void PhysicsList::SetCuts()
{
    // Production cuts
    SetCutValue(0.01*mm, "gamma");
    SetCutValue(0.01*mm, "e-");
    SetCutValue(0.01*mm, "e+");
    SetCutValue(0.01*mm, "proton");
    // No cut on neutrons – tracked by HP processes

    if (verboseLevel > 0) DumpCutValuesTable();
}
