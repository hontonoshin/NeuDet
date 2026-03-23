//============================================================
// main.cc  –  Li6ZnS:Ag neutron detector simulation
// Run: ./neutron_sim run.mac
//============================================================
#include "DetectorConstruction.hh"
#include "PhysicsList.hh"
#include "ActionInitialization.hh"

#include "G4RunManagerFactory.hh"
#include "G4UImanager.hh"
#include "G4UIExecutive.hh"
#include "G4VisExecutive.hh"
#include "Randomize.hh"
#include "G4SystemOfUnits.hh"

int main(int argc, char** argv)
{
    // Random seed
    G4Random::setTheEngine(new CLHEP::RanecuEngine);
    G4Random::setTheSeed(time(nullptr));

    // Run manager
    auto* runManager = G4RunManagerFactory::CreateRunManager(
                           G4RunManagerType::SerialOnly);

    auto* det = new DetectorConstruction();
    runManager->SetUserInitialization(det);
    runManager->SetUserInitialization(new PhysicsList());
    runManager->SetUserInitialization(new ActionInitialization(det));

    runManager->Initialize();

    G4UImanager* UImanager = G4UImanager::GetUIpointer();

    if (argc > 1) {
        // Batch mode
        G4String cmd = "/control/execute ";
        UImanager->ApplyCommand(cmd + argv[1]);
    } else {
        // Interactive
        auto* ui = new G4UIExecutive(argc, argv);
        ui->SessionStart();
        delete ui;
    }

    delete runManager;
    return 0;
}
