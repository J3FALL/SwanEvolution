pid = 1
parset_id = 293
mod_id = 0

for (mod_id in c(0, 1, 2, 3, 4, 5, 6, 7, 15, 16, 17, 18, 25, 26))
{
    print(glue("MODEL {mod_id}"))
    for (pid in 1 : 9)
    {
        print(glue("POINT {pid}"))
        ww1_obs = read.csv(
        glue(
        "//samples/ww-res/obs_fromww_{pid}.csv"
        ),
        header = TRUE,
        sep = ","
        )[1 : 253,]

        for (parset_id in 1 : 980)
        {
            csv_ref = read.csv(
            glue(
            "\\wind-noise-runs\\results-new\\{mod_id}\\K{pid}a_ns{mod_id}_run{parset_id}.tab"
            ),
            header = FALSE,
            sep = "",
            skip = 7,
            as.is = TRUE
            )

            csv_0 = read.csv(
            glue(
            "\\wind-noise-runs\\results-new\\{0}\\K{pid}a_ns0_run{parset_id}.tab"
            ),
            header = FALSE,
            sep = "",
            skip = 7
            )

            ref_forecst = csv_0$V2

            ens_for_cor = csv_ref$V2
            ens_for = csv_ref$V2

            if (mod_id != 0)
            {
                ww = ww1_obs$hs

                peak_cond1 = mean(ww1_obs$hs) * 0.9

                bad_peak_cond = (ww < peak_cond1 &
                ens_for_cor > peak_cond1) |
                ens_for_cor == 0

                ens_for_cor[bad_peak_cond] = ref_forecst[bad_peak_cond]

                peak_cond2 = mean(ww) * 0.9
                miss_peak_cond = ens_for_cor < ref_forecst &
                ww > peak_cond2

                ens_for_cor[miss_peak_cond] = ref_forecst[miss_peak_cond]

                high_peak_cond = ww1_obs$hs > peak_cond1 &
                ens_for_cor / ww > 1.7
                ens_for_cor[high_peak_cond] = ens_for_cor[high_peak_cond] / 1.7

                low_peak_cond = ww1_obs$hs > peak_cond1 &
                ens_for_cor / ww < 0.7
                ens_for_cor[low_peak_cond] = ens_for_cor[low_peak_cond] / 0.7
            }

            csv_new = csv_ref
            csv_new$V2 = round(ens_for_cor, 5)


            write.csv(
            csv_new,
            glue(
            "wind-postproc\\out\\K{pid}a_ns{mod_id}_run{parset_id}_postproc.tab"
            ),
            quote = FALSE,
            row.names = FALSE,
            col.names = FALSE
            )
        }
    }
}
