""""
Mapping between dataset columns names and LaTeX
"""

# A1 = Bottom
# A2 = Top
label_dict = {
    "MAE (meV)": r"MAE$_\mathrm{SC}$",
    "BG_ind_up": r"$BG_\mathrm{ind}^{\uparrow}$",
    "BG_ind_down": r"$BG_\mathrm{ind}^{\downarrow}$",
    "BG_dir_up": r"$BG_\mathrm{dir}^{\uparrow}$",
    "BG_dir_down": r"$BG_\mathrm{dir}^{\downarrow}$",
    "dimer_len": r"$d_{len}$",
    "h_substrate": r"$h_{sub}$",
    "dimer_defect_dist": "dimerdefect",
    "tilt": "tilt",
    "sr_A1_dxyUP": r"$DOS_{B, d_{xy}}^{\uparrow}$",
    "sr_A2_dxyUP": r"$DOS_{T, d_{xy}}^{\uparrow}$",
    "sr_A1_dxyDOWN": r"$DOS_{B, d_{xy}}^{\downarrow}$",
    "sr_A2_dxyDOWN": r"$DOS_{T, d_{xy}}^{\downarrow}$",
    "sr_A1_dyzUP": r"$DOS_{B, d_{yz}}^{\uparrow}$",
    "sr_A2_dyzUP": r"$DOS_{T, d_{yz}}^{\uparrow}$",
    "sr_A1_dyzDOWN": r"$DOS_{B, d_{yz}}^{\downarrow}$",
    "sr_A2_dyzDOWN": r"$DOS_{T, d_{yz}}^{\downarrow}$",
    "sr_A1_dz2-r2UP": r"$DOS_{B, d_{z^2}}^{\uparrow}$",
    "sr_A2_dz2-r2UP": r"$DOS_{T, d_{z^2}}^{\uparrow}$",
    "sr_A1_dz2-r2DOWN": r"$DOS_{B, d_{z^2}}^{\downarrow}$",
    "sr_A2_dz2-r2DOWN": r"$DOS_{T, d_{z^2}}^{\downarrow}$",
    "sr_A1_dxzUP": r"$DOS_{B, d_{xz}}^{\uparrow}$",
    "sr_A2_dxzUP": r"$DOS_{T, d_{xz}}^{\uparrow}$",
    "sr_A1_dxzDOWN": r"$DOS_{B, d_{xz}}^{\downarrow}$",
    "sr_A2_dxzDOWN": r"$DOS_{T, d_{xz}}^{\downarrow}$",
    "sr_A1_dx2-y2UP": r"$DOS_{B, d_{x^2-y^2}}^{\uparrow}$",
    "sr_A2_dx2-y2UP": r"$DOS_{T, d_{x^2-y^2}}^{\uparrow}$",
    "sr_A1_dx2-y2DOWN": r"$DOS_{B, d_{x^2-y^2}}^{\downarrow}$",
    "sr_A2_dx2-y2DOWN": r"$DOS_{T, d_{x^2-y^2}}^{\uparrow}$",
    "sr_A1_magx": r"$\mu_B$",
    "sr_A2_magx": r"$\mu_T$",
    "sr_A1_dosmodeldxyUP_integral_below_1.0": r"$I_{B, d_{xy}}^{\uparrow -}$",
    "sr_A1_dosmodeldyzUP_integral_below_1.0": r"$I_{B, d_{yz}}^{\uparrow -}$",
    "sr_A1_dosmodeldz2-r2UP_integral_below_1.0": r"$I_{B, d_{z^2}}^{\uparrow -}$",
    "sr_A1_dosmodeldxzUP_integral_below_1.0": r"$I_{B, d_{xz}}^{\uparrow -}$",
    "sr_A1_dosmodeldx2-y2UP_integral_below_1.0": r"$I_{B, d_{x^2-y^2}}^{\uparrow -}$",
    "sr_A1_dosmodeldxyUP_integral_above_1.0": r"$I_{B, d_{xy}}^{\uparrow +}$",
    "sr_A1_dosmodeldyzUP_integral_above_1.0": r"$I_{B, d_{yz}}^{\uparrow +}$",
    "sr_A1_dosmodeldz2-r2UP_integral_above_1.0": r"$I_{B, d_{z^2}}^{\uparrow +}$",
    "sr_A1_dosmodeldxzUP_integral_above_1.0": r"$I_{B, d_{xz}}^{\uparrow +}$",
    "sr_A1_dosmodeldx2-y2UP_integral_above_1.0": r"$I_{B, d_{x^2-y^2}}^{\uparrow +}$",
    "sr_A1_dosmodeldxyDOWN_integral_below_1.0": r"$I_{B, d_{xy}}^{\downarrow -}$",
    "sr_A1_dosmodeldyzDOWN_integral_below_1.0": r"$I_{B, d_{yz}}^{\downarrow -}$",
    "sr_A1_dosmodeldz2-r2DOWN_integral_below_1.0": r"$I_{B, d_{z^2}}^{\downarrow -}$",
    "sr_A1_dosmodeldxzDOWN_integral_below_1.0": r"$I_{B, d_{xz}}^{\downarrow -}$",
    "sr_A1_dosmodeldx2-y2DOWN_integral_below_1.0": r"$I_{B, d_{x^2-y^2}}^{\downarrow -}$",
    "sr_A1_dosmodeldxyDOWN_integral_above_1.0": r"$I_{B, d_{xy}}^{\downarrow +}$",
    "sr_A1_dosmodeldyzDOWN_integral_above_1.0": r"$I_{B, d_{yz}}^{\downarrow +}$",
    "sr_A1_dosmodeldz2-r2DOWN_integral_above_1.0": r"$I_{B, d_{z^2}}^{\downarrow +}$",
    "sr_A1_dosmodeldxzDOWN_integral_above_1.0": r"$I_{B, d_{xz}}^{\downarrow +}$",
    "sr_A1_dosmodeldx2-y2DOWN_integral_above_1.0": r"$I_{B, d_{x^2-y^2}}^{\downarrow +}$",
    "sr_A2_dosmodeldxyUP_integral_below_1.0": r"$I_{T, d_{xy}}^{\uparrow -}$",
    "sr_A2_dosmodeldyzUP_integral_below_1.0": r"$I_{T, d_{yz}}^{\uparrow -}$",
    "sr_A2_dosmodeldz2-r2UP_integral_below_1.0": r"$I_{T, d_{z^2}}^{\uparrow -}$",
    "sr_A2_dosmodeldxzUP_integral_below_1.0": r"$I_{T, d_{xz}}^{\uparrow -}$",
    "sr_A2_dosmodeldx2-y2UP_integral_below_1.0": r"$I_{T, d_{x^2-y^2}}^{\uparrow -}$",
    "sr_A2_dosmodeldxyUP_integral_above_1.0": r"$I_{T, d_{xy}}^{\uparrow +}$",
    "sr_A2_dosmodeldyzUP_integral_above_1.0": r"$I_{T, d_{yz}}^{\uparrow +}$",
    "sr_A2_dosmodeldz2-r2UP_integral_above_1.0": r"$I_{T, d_{z^2}}^{\uparrow +}$",
    "sr_A2_dosmodeldxzUP_integral_above_1.0": r"$I_{T, d_{xz}}^{\uparrow +}$",
    "sr_A2_dosmodeldx2-y2UP_integral_above_1.0": r"$I_{T, d_{x^2-y^2}}^{\uparrow +}$",
    "sr_A2_dosmodeldxyDOWN_integral_below_1.0": r"$I_{T, d_{xy}}^{\downarrow -}$",
    "sr_A2_dosmodeldyzDOWN_integral_below_1.0": r"$I_{T, d_{yz}}^{\downarrow -}$",
    "sr_A2_dosmodeldz2-r2DOWN_integral_below_1.0": r"$I_{T, d_{z^2}}^{\downarrow -}$",
    "sr_A2_dosmodeldxzDOWN_integral_below_1.0": r"$I_{T, d_{xz}}^{\downarrow -}$",
    "sr_A2_dosmodeldx2-y2DOWN_integral_below_1.0": r"$I_{T, d_{x^2-y^2}}^{\downarrow -}$",
    "sr_A2_dosmodeldxyDOWN_integral_above_1.0": r"$I_{T, d_{xy}}^{\downarrow +}$",
    "sr_A2_dosmodeldyzDOWN_integral_above_1.0": r"$I_{T, d_{yz}}^{\downarrow +}$",
    "sr_A2_dosmodeldz2-r2DOWN_integral_above_1.0": r"$I_{T, d_{z^2}}^{\downarrow +}$",
    "sr_A2_dosmodeldxzDOWN_integral_above_1.0": r"$I_{T, d_{xz}}^{\downarrow +}$",
    "sr_A2_dosmodeldx2-y2DOWN_integral_above_1.0": r"$I_{T, d_{x^2-y^2}}^{\downarrow +}$",
    "pot_A1": r"$\Phi$",
    "bader_A1": r"$q_B$",
    "bader_A2": r"$q_T$",
    "sr_A1_dosmodeldxyUP_E_below": r"$E_{B, d_{xy}}^{\uparrow -}$",
    "sr_A1_dosmodeldxyUP_peak_below": r"$D_{B, d_{xy}}^{\uparrow -}$",
    "sr_A1_dosmodeldyzUP_E_below": r"$E_{B, d_{yz}}^{\uparrow -}$",
    "sr_A1_dosmodeldyzUP_peak_below": r"$D_{B, d_{yz}}^{\uparrow -}$",
    "sr_A1_dosmodeldz2-r2UP_E_below": r"$E_{B, d_{z^2}}^{\uparrow -}$",
    "sr_A1_dosmodeldz2-r2UP_peak_below": r"$D_{B, d_{z^2}}^{\uparrow -}$",
    "sr_A1_dosmodeldxzUP_E_below": r"$E_{B, d_{xz}}^{\uparrow -}$",
    "sr_A1_dosmodeldxzUP_peak_below": r"$D_{B, d_{xz}}^{\uparrow -}$",
    "sr_A1_dosmodeldx2-y2UP_E_below": r"$E_{B, d_{x^2-y^2}}^{\uparrow -}$",
    "sr_A1_dosmodeldx2-y2UP_peak_below": r"$D_{B, d_{x^2-y^2}}^{\uparrow -}$",
    "sr_A1_dosmodeldxyUP_E_above": r"$E_{B, d_{xy}}^{\uparrow +}$",
    "sr_A1_dosmodeldxyUP_peak_above": r"$D_{B, d_{xy}}^{\uparrow +}$",
    "sr_A1_dosmodeldyzUP_E_above": r"$E_{B, d_{yz}}^{\uparrow +}$",
    "sr_A1_dosmodeldyzUP_peak_above": r"$D_{B, d_{yz}}^{\uparrow +}$",
    "sr_A1_dosmodeldz2-r2UP_E_above": r"$E_{B, d_{z^2}}^{\uparrow +}$",
    "sr_A1_dosmodeldz2-r2UP_peak_above": r"$D_{B, d_{z^2}}^{\uparrow +}$",
    "sr_A1_dosmodeldxzUP_E_above": r"$E_{B, d_{xz}}^{\uparrow +}$",
    "sr_A1_dosmodeldxzUP_peak_above": r"$D_{B, d_{xz}}^{\uparrow +}$",
    "sr_A1_dosmodeldx2-y2UP_E_above": r"$E_{B, d_{x^2-y^2}}^{\uparrow +}$",
    "sr_A1_dosmodeldx2-y2UP_peak_above": r"$D_{B, d_{x^2-y^2}}^{\uparrow +}$",
    "sr_A1_dosmodeldxyDOWN_E_below": r"$E_{B, d_{xy}}^{\downarrow -}$",
    "sr_A1_dosmodeldxyDOWN_peak_below": r"$D_{B, d_{xy}}^{\downarrow -}$",
    "sr_A1_dosmodeldyzDOWN_E_below": r"$E_{B, d_{yz}}^{\downarrow -}$",
    "sr_A1_dosmodeldyzDOWN_peak_below": r"$D_{B, d_{yz}}^{\downarrow -}$",
    "sr_A1_dosmodeldz2-r2DOWN_E_below": r"$E_{B, d_{z^2}}^{\downarrow -}$",
    "sr_A1_dosmodeldz2-r2DOWN_peak_below": r"$D_{B, d_{z^2}}^{\downarrow -}$",
    "sr_A1_dosmodeldxzDOWN_E_below": r"$E_{B, d_{xz}}^{\downarrow -}$",
    "sr_A1_dosmodeldxzDOWN_peak_below": r"$D_{B, d_{xz}}^{\downarrow -}$",
    "sr_A1_dosmodeldx2-y2DOWN_E_below": r"$E_{B, d_{x^2-y^2}}^{\downarrow -}$",
    "sr_A1_dosmodeldx2-y2DOWN_peak_below": r"$D_{B, d_{x^2-y^2}}^{\downarrow -}$",
    "sr_A1_dosmodeldxyDOWN_E_above": r"$E_{B, d_{xy}}^{\downarrow +}$",
    "sr_A1_dosmodeldxyDOWN_peak_above": r"$D_{B, d_{xy}}^{\downarrow +}$",
    "sr_A1_dosmodeldyzDOWN_E_above": r"$E_{B, d_{yz}}^{\downarrow +}$",
    "sr_A1_dosmodeldyzDOWN_peak_above": r"$D_{B, d_{yz}}^{\downarrow +}$",
    "sr_A1_dosmodeldz2-r2DOWN_E_above": r"$E_{B, d_{z^2}}^{\downarrow +}$",
    "sr_A1_dosmodeldz2-r2DOWN_peak_above": r"$D_{B, d_{z^2}}^{\downarrow +}$",
    "sr_A1_dosmodeldxzDOWN_E_above": r"$E_{B, d_{xz}}^{\downarrow +}$",
    "sr_A1_dosmodeldxzDOWN_peak_above": r"$D_{B, d_{xz}}^{\downarrow +}$",
    "sr_A1_dosmodeldx2-y2DOWN_E_above": r"$E_{B, d_{x^2-y^2}^{\downarrow +}$",
    "sr_A1_dosmodeldx2-y2DOWN_peak_above": r"$D_{B, d_{x^2-y^2}}^{\downarrow +}$",
    "sr_A2_dosmodeldxyUP_E_below": r"$E_{T, d_{xy}}^{\uparrow -}$",
    "sr_A2_dosmodeldxyUP_peak_below": r"$D_{T, d_{xy}}^{\uparrow -}$",
    "sr_A2_dosmodeldyzUP_E_below": r"$E_{T, d_{yz}}^{\uparrow -}$",
    "sr_A2_dosmodeldyzUP_peak_below": r"$D_{T, d_{yz}}^{\uparrow -}$",
    "sr_A2_dosmodeldz2-r2UP_E_below": r"$E_{T, d_{z^2}}^{\uparrow -}$",
    "sr_A2_dosmodeldz2-r2UP_peak_below": r"$D_{T, d_{z^2}}^{\uparrow -}$",
    "sr_A2_dosmodeldxzUP_E_below": r"$E_{T, d_{xz}}^{\uparrow -}$",
    "sr_A2_dosmodeldxzUP_peak_below": r"$D_{T, d_{xz}}^{\uparrow -}$",
    "sr_A2_dosmodeldx2-y2UP_E_below": r"$E_{T, d_{x^2-y^2}}^{\uparrow -}$",
    "sr_A2_dosmodeldx2-y2UP_peak_below": r"$D_{T, d_{x^2-y^2}}^{\uparrow -}$",
    "sr_A2_dosmodeldxyUP_E_above": r"$E_{T, d_{xy}}^{\uparrow +}$",
    "sr_A2_dosmodeldxyUP_peak_above": r"$D_{T, d_{xy}}^{\uparrow +}$",
    "sr_A2_dosmodeldyzUP_E_above": r"$E_{T, d_{yz}}^{\uparrow +}$",
    "sr_A2_dosmodeldyzUP_peak_above": r"$D_{T, d_{yz}}^{\uparrow +}$",
    "sr_A2_dosmodeldz2-r2UP_E_above": r"$E_{T, d_{z^2}}^{\uparrow +}$",
    "sr_A2_dosmodeldz2-r2UP_peak_above": r"$D_{T, d_{z^2}}^{\uparrow +}$",
    "sr_A2_dosmodeldxzUP_E_above": r"$E_{T, d_{xz}}^{\uparrow +}$",
    "sr_A2_dosmodeldxzUP_peak_above": r"$D_{T, d_{xz}}^{\uparrow +}$",
    "sr_A2_dosmodeldx2-y2UP_E_above": r"$E_{T, d_{x^2-y^2}}^{\uparrow +}$",
    "sr_A2_dosmodeldx2-y2UP_peak_above": r"$D_{T, d_{x^2-y^2}}^{\uparrow +}$",
    "sr_A2_dosmodeldxyDOWN_E_below": r"$E_{T, d_{xy}}^{\downarrow -}$",
    "sr_A2_dosmodeldxyDOWN_peak_below": r"$D_{T, d_{xy}}^{\downarrow -}$",
    "sr_A2_dosmodeldyzDOWN_E_below": r"$E_{T, d_{yz}}^{\downarrow -}$",
    "sr_A2_dosmodeldyzDOWN_peak_below": r"$D_{T, d_{yz}}^{\downarrow -}$",
    "sr_A2_dosmodeldz2-r2DOWN_E_below": r"$E_{T, d_{z^2}}^{\downarrow -}$",
    "sr_A2_dosmodeldz2-r2DOWN_peak_below": r"$D_{T, d_{z^2}}^{\downarrow -}$",
    "sr_A2_dosmodeldxzDOWN_E_below": r"$E_{T, d_{xz}}^{\downarrow -}$",
    "sr_A2_dosmodeldxzDOWN_peak_below": r"$D_{T, d_{xz}}^{\downarrow -}$",
    "sr_A2_dosmodeldx2-y2DOWN_E_below": r"$E_{T, d_{x^2-y^2}}^{\downarrow -}$",
    "sr_A2_dosmodeldx2-y2DOWN_peak_below": r"$D_{T, d_{x^2-y^2}}^{\downarrow -}$",
    "sr_A2_dosmodeldxyDOWN_E_above": r"$E_{T, d_{xy}}^{\downarrow +}$",
    "sr_A2_dosmodeldxyDOWN_peak_above": r"$D_{T, d_{xy}}^{\downarrow +}$",
    "sr_A2_dosmodeldyzDOWN_E_above": r"$E_{T, d_{yz}}^{\downarrow +}$",
    "sr_A2_dosmodeldyzDOWN_peak_above": r"$D_{T, d_{yz}}^{\downarrow +}$",
    "sr_A2_dosmodeldz2-r2DOWN_E_above": r"$E_{T, d_{z^2}}^{\downarrow +}$",
    "sr_A2_dosmodeldz2-r2DOWN_peak_above": r"$D_{T, d_{z^2}}^{\downarrow +}$",
    "sr_A2_dosmodeldxzDOWN_E_above": r"$E_{T, d_{xz}}^{\downarrow +}$",
    "sr_A2_dosmodeldxzDOWN_peak_above": r"$D_{T, d_{xz}}^{\downarrow +}$",
    "sr_A2_dosmodeldx2-y2DOWN_E_above": r"$E_{T, d_{x^2-y^2}}^{\downarrow +}$",
    "sr_A2_dosmodeldx2-y2DOWN_peak_above": r"$D_{T, d_{x^2-y^2}}^{\downarrow +}$",
}


def labels(txt):
    if txt in label_dict:
        return label_dict[txt]
    else:
        return txt
