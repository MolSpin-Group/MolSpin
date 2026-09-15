// Synthetic spin-1/2 reference: g=2, 20 K, no hyperfine interaction.
// Scan 320-360 mT at 9.5 GHz. Gaussian absorption FWHM: 1 mT.
SpinSystem reference {
 Spin E { type=electron; spin=1/2; tensor=isotropic(2); }
 Interaction B0 {
   type=zeeman; spins=E; field="0 0 0.32";
   commonprefactor=true; prefactor=1;
 }
 Properties populations {
   initialstate=Thermal; frame=eigen;
   thermalhamiltonian=B0; temperature=20;
 }
}
Settings {
 Settings general { steps=401; outputprecision=17; }
 Action sweep {
   type=AddVector; vector=reference.B0.field;
   direction="0 0 1"; value=0.0001;
 }
}
Run {
 Task spectrum {
   type=ResonanceGeneral; solver=exact;
   mwfrequency=9.5; linewidth=1; lineshape=gaussian;
   harmonic=0; // Use 1 for the first field derivative.
   detectspins=E; fieldinteraction=B0; hamiltonianh0list=B0;
   powdergridtype=sophe; powdergridsymmetry=Dinfh;
   powdergridsize=19; powdergammapoints=1;
   sweepcache=true; sweepcachemode=exact;
   datafile="SpinHalf_X_band.dat"; appenddata=false;
 }
}
