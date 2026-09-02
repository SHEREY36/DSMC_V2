# Legacy conditional-GMM audit

The legacy loader defines its training response as
`Et_f / (Et_f + Er1_f + Er2_f)` using the inelastic post-collision energies.
The current v1 runtime subsequently draws the BL scalar loss and subtracts it
again through modal routing. Consequently the GMM already contains an
inelastic-energy effect that is composed with a second, independent loss in
the runtime. The variational production mode therefore does not load or call
the conditional GMM. It transfers only the surviving CTC energy partition;
the frozen BL scalar law remains authoritative for total loss. The complete
legacy path is retained solely for A/B comparison.
