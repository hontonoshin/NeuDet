//============================================================
// PrimaryGeneratorAction.cc
//
// Reactor neutron spectrum composed of three regions following:
//   [1] Lamarsh & Baratta, "Introduction to Nuclear Engineering",
//       3rd ed. (2001), Ch.5
//   [2] Knoll G.F., "Radiation Detection and Measurement",
//       4th ed. (2010), Appendix C
//   [3] IAEA-TECDOC-1234 (2001) – Research reactor spectrum models
//
// Spectrum model:
//   φ(E) = φ_thermal(E) + φ_epithermal(E) + φ_fast(E)
//
//   Thermal  (E < 0.5 eV):   Maxwell-Boltzmann at T=600 K (hot reactor)
//     φ_th(E) ∝ E · exp(-E/kT),  kT = 0.0517 eV
//
//   Epithermal (0.5 eV – 0.1 MeV):  1/E (Fermi slowing-down)
//     φ_ep(E) ∝ 1/E
//
//   Fast (0.1 MeV – 10 MeV):  Watt fission spectrum
//     φ_f(E) ∝ exp(-E/a) · sinh(sqrt(b·E))
//     U-235: a=0.988 MeV, b=2.249/MeV  (ENDF/B-VIII.0)
//
//   Relative normalisation (flux fractions):
//     Thermal    : 0.50
//     Epithermal : 0.25
//     Fast       : 0.25
//
// Source: pencil beam (point source) on detector face
//============================================================
#include "PrimaryGeneratorAction.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4ParticleDefinition.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include "Randomize.hh"
#include <cmath>
#include <stdexcept>

namespace {
    // Watt fission spectrum parameters (235U, ENDF/B-VIII.0)
    constexpr double a_watt = 0.988;   // MeV
    constexpr double b_watt = 2.249;   // 1/MeV

    // Maxwell-Boltzmann temperature (600 K hot-zero-power)
    constexpr double kT_MeV = 0.0517e-3; // MeV   (kT = k_B × 600 K)

    // Flux fraction weights
    constexpr double w_thermal     = 0.50;
    constexpr double w_epithermal  = 0.25;
    // w_fast = 1 - w_thermal - w_epithermal = 0.25

    // Energy boundaries
    constexpr double E_th_max  = 0.5e-6;   // 0.5 eV in MeV
    constexpr double E_ep_max  = 0.1;      // 100 keV in MeV
    constexpr double E_fast_max = 10.0;    // 10 MeV

    double MaxwellBoltzmann(double E_MeV) {
        // φ ∝ E·exp(-E/kT)
        return E_MeV * std::exp(-E_MeV / kT_MeV);
    }

    double WattFission(double E_MeV) {
        if (E_MeV <= 0.0) return 0.0;
        return std::exp(-E_MeV / a_watt) * std::sinh(std::sqrt(b_watt * E_MeV));
    }
}

PrimaryGeneratorAction::PrimaryGeneratorAction()
  : G4VUserPrimaryGeneratorAction(),
    fParticleGun(nullptr),
    fCurrentEnergy(0.0)
{
    fParticleGun = new G4ParticleGun(1);

    G4ParticleTable* table = G4ParticleTable::GetParticleTable();
    G4ParticleDefinition* neutron = table->FindParticle("neutron");
    fParticleGun->SetParticleDefinition(neutron);

    // Direction: +Z (into detector)
    fParticleGun->SetParticleMomentumDirection(G4ThreeVector(0, 0, 1));

    // Position: just upstream of the detector (z = -26.1 cm)
    fParticleGun->SetParticlePosition(G4ThreeVector(0, 0, -12.0*cm));

    BuildReactorSpectrumCDF();
}

PrimaryGeneratorAction::~PrimaryGeneratorAction()
{
    delete fParticleGun;
}

void PrimaryGeneratorAction::BuildReactorSpectrumCDF()
{
    // Build CDF on a log-uniform grid: 1e-10 MeV – 10 MeV
    // Sampling variable: u = log(E), so the Jacobian for each bin is E·dlogE.
    //
    // ── BUG FIX (critical) ──────────────────────────────────────────────────
    // The original code multiplied the already-computed φ(E) by E a second
    // time as the "Jacobian", but φ_thermal(E) = E·exp(-E/kT) already
    // contains one factor of E.  This made the thermal PDF proportional to
    // E²·exp(-E/kT), whose peak value at E=kT is ~kT² ≈ 2.7×10⁻⁹ — roughly
    // 4.5×10¹² times smaller than the Watt-spectrum peak near 1 MeV.
    // As a result, <0.001% of primaries were thermal instead of the intended
    // 50%, and the thermal peak was entirely absent from output spectra.
    //
    // ── Correct approach ────────────────────────────────────────────────────
    // We sample in log(E) space.  The CDF integrand is:
    //
    //   g(E) = A_k · f_k(E) · E          (Jacobian = E for d(logE) → dE)
    //
    // where f_k(E) is the *un-normalised* shape of component k, and A_k is
    // chosen so that  ∫ A_k · f_k(E) · E · d(logE)  = w_k  (desired fraction).
    //
    // We compute A_k = w_k / I_k  where  I_k = ∫ f_k(E)·E d(logE)
    // integrated numerically over the k-th energy range on the same grid.
    // ────────────────────────────────────────────────────────────────────────

    const int    N       = 4000;
    const double E_min   = 1.0e-10;   // 0.1 meV
    const double E_max   = 10.0;      // 10 MeV
    const double logEmin = std::log(E_min);
    const double logEmax = std::log(E_max);
    const double dlogE   = (logEmax - logEmin) / N;

    fEnergyCDF.resize(N+1);
    std::vector<double> f_shape(N+1, 0.0);  // f_k(E) — un-normalised shape

    // Step 1: fill energy grid and raw shape values
    for (int i = 0; i <= N; ++i) {
        double E = std::exp(logEmin + i * dlogE);
        fEnergyCDF[i] = E;

        if (E <= E_th_max) {
            // Maxwell-Boltzmann shape:  f(E) = E · exp(-E/kT)
            f_shape[i] = MaxwellBoltzmann(E);           // = E·exp(-E/kT)
        } else if (E <= E_ep_max) {
            // Fermi 1/E slowing-down shape:  f(E) = 1/E
            f_shape[i] = 1.0 / E;
        } else {
            // Watt fission spectrum:  f(E) = exp(-E/a)·sinh(√(b·E))
            f_shape[i] = WattFission(E);
        }
    }

    // Step 2: compute per-component integrals I_k = ∫ f_k(E)·E d(logE)
    //         using the trapezoidal rule on the log-uniform grid.
    double I_th = 0.0, I_ep = 0.0, I_f = 0.0;
    for (int i = 1; i <= N; ++i) {
        double E0 = fEnergyCDF[i-1], E1 = fEnergyCDF[i];
        double g0 = f_shape[i-1] * E0;
        double g1 = f_shape[i]   * E1;
        double contrib = 0.5 * (g0 + g1) * dlogE;

        if (E1 <= E_th_max)
            I_th += contrib;
        else if (E1 <= E_ep_max)
            I_ep += contrib;
        else
            I_f  += contrib;
    }

    G4cout << "\n=== Reactor spectrum normalisation check ===" << G4endl;
    G4cout << "  I_thermal   = " << I_th << G4endl;
    G4cout << "  I_epithermal= " << I_ep << G4endl;
    G4cout << "  I_fast      = " << I_f  << G4endl;

    // Step 3: scaling factors A_k = w_k / I_k
    const double A_th = (I_th > 0) ? w_thermal    / I_th : 0.0;
    const double A_ep = (I_ep > 0) ? w_epithermal / I_ep : 0.0;
    const double A_f  = (I_f  > 0) ? (1.0 - w_thermal - w_epithermal) / I_f : 0.0;

    // Step 4: build correctly-weighted PDF = A_k · f_k(E) · E
    std::vector<double> pdf(N+1, 0.0);
    for (int i = 0; i <= N; ++i) {
        double E = fEnergyCDF[i];
        double A = 0.0;
        if      (E <= E_th_max) A = A_th;
        else if (E <= E_ep_max) A = A_ep;
        else                    A = A_f;
        pdf[i] = A * f_shape[i] * E;   // integrand in log(E) space
    }

    // Step 5: integrate to CDF, then normalise
    fCDF.resize(N+1, 0.0);
    for (int i = 1; i <= N; ++i)
        fCDF[i] = fCDF[i-1] + 0.5*(pdf[i-1] + pdf[i]) * dlogE;

    double total = fCDF[N];
    if (total > 0)
        for (auto& c : fCDF) c /= total;

    // Verification printout
    double f_th_check = 0., f_ep_check = 0., f_f_check = 0.;
    for (int i = 1; i <= N; ++i) {
        double dc = fCDF[i] - fCDF[i-1];
        double E  = fEnergyCDF[i];
        if      (E <= E_th_max) f_th_check += dc;
        else if (E <= E_ep_max) f_ep_check += dc;
        else                    f_f_check  += dc;
    }
    G4cout << "  Sampled fractions (should be 50/25/25%):" << G4endl;
    G4cout << "    Thermal     : " << f_th_check*100. << " %" << G4endl;
    G4cout << "    Epithermal  : " << f_ep_check*100. << " %" << G4endl;
    G4cout << "    Fast        : " << f_f_check*100.  << " %" << G4endl;
    G4cout << "============================================\n" << G4endl;
}

G4double PrimaryGeneratorAction::SampleReactorEnergy()
{
    double rnd = G4UniformRand();

    // Binary search in CDF
    int lo = 0, hi = (int)fCDF.size()-1;
    while (hi - lo > 1) {
        int mid = (lo + hi) / 2;
        if (fCDF[mid] <= rnd) lo = mid;
        else                  hi = mid;
    }

    // Linear interpolation
    double f = (fCDF[hi] - fCDF[lo] > 1e-14)
               ? (rnd - fCDF[lo]) / (fCDF[hi] - fCDF[lo])
               : 0.5;
    double E = fEnergyCDF[lo] + f*(fEnergyCDF[hi] - fEnergyCDF[lo]);
    return E;  // MeV
}

void PrimaryGeneratorAction::GeneratePrimaries(G4Event* event)
{
    fCurrentEnergy = SampleReactorEnergy();
    fParticleGun->SetParticleEnergy(fCurrentEnergy * MeV);
    fParticleGun->GeneratePrimaryVertex(event);
}
