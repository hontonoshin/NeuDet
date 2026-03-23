//============================================================
// PrimaryGeneratorAction.hh
// Reactor neutron spectrum (Maxwell + epithermal + fast)
// Based on: IAEA-TECDOC-1234 and Keepin (1965) spectra
//============================================================
#ifndef PrimaryGeneratorAction_h
#define PrimaryGeneratorAction_h 1

#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4ParticleGun.hh"
#include "globals.hh"
#include <vector>

class G4Event;

class PrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction
{
public:
    PrimaryGeneratorAction();
    ~PrimaryGeneratorAction() override;

    void GeneratePrimaries(G4Event* event) override;

    G4double GetParticleEnergy() const { return fCurrentEnergy; }

private:
    G4ParticleGun* fParticleGun;
    G4double       fCurrentEnergy;

    // Tabulated reactor spectrum CDF for rejection sampling
    void BuildReactorSpectrumCDF();
    G4double SampleReactorEnergy();

    std::vector<G4double> fEnergyCDF;   // energies [MeV]
    std::vector<G4double> fCDF;         // cumulative probabilities
};

#endif
