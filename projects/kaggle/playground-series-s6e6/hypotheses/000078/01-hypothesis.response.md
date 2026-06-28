{
  "title": "APO airmass chromatic calibration",
  "group_name": "apo_airmass_chromaticity",
  "family": "instrumental_calibration",
  "summary": "Represent each object by the wavelength-dependent atmospheric extinction geometry implied by its declination at the Apache Point SDSS telescope, so subtle color shifts can be separated from intrinsic stellar, galaxy, and quasar SED differences.",
  "depends_on": [],
  "strategy": "Use fixed APO latitude 32.78036 deg and SDSS reference airmass 1.3; for each row compute z_min=abs(delta-32.78036), X=1/cos(z_min*pi/180) clipped to [1.0,1.6], dX=X-1.3, fixed extinction coefficients k=[0.48,0.18,0.10,0.07,0.05] for u,g,r,i,z, predicted band offsets e_b=k_b*dX, adjacent predicted color shifts e_u-e_g through e_i-e_z, signed projection of observed adjacent color vector [u-g,g-r,r-i,i-z] onto the extinction-color vector [k_u-k_g,k_g-k_r,k_r-k_i,k_i-k_z], orthogonal residual norm from that projection, blue-sensitive amplification using (u-g)*(X-1.3) and (g-r)*(X-1.3), and hard airmass bins X<1.15, 1.15<=X<1.30, 1.30<=X<1.45, X>=1.45; clip delta to [-90,90], replace nonfinite trigonometric results with X=1.3, and use epsilon 1e-9 in vector normalizations. Source anchors: https://www.sdss4.org/dr17/algorithms/fluxcal/ and https://www.sdss.org/instruments/.",
  "expected_signal": "Quasar-star and galaxy-star boundaries often depend on small ugriz color offsets, especially in u and g, while SDSS photometry is calibrated around a reference atmospheric airmass; exposing the physically expected chromatic shift may help clarify borderline blue objects and declination-linked calibration residuals without using labels.",
  "risk": "SDSS already corrects atmospheric extinction, so the residual signal may be weak or redundant with raw delta and prior sky-position features; fixed extinction coefficients approximate actual nightly conditions and could overfit survey-footprint artifacts if train and test declination coverage differ."
}
