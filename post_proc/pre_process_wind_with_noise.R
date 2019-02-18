Sys.setenv(TZ = 'UTC')

set.seed(42)

library(data.table)
library(glue)

# Multiple plot function
#
# ggplot objects can be passed in ..., or to plotlist (as a list of ggplot objects)
# - cols:   Number of columns in layout
# - layout: A matrix specifying the layout. If present, 'cols' is ignored.
#
# If the layout is something like matrix(c(1,2,3,3), nrow=2, byrow=TRUE),
# then plot 1 will go in the upper left, 2 will go in the upper right, and
# 3 will go all the way across the bottom.
#
multiplot <-
function(...,
plotlist = NULL,
file,
cols = 1,
layout = NULL) {
    library(grid)

    # Make a list from the ... arguments and plotlist
    plots <- c(list(...), plotlist)

    numPlots = length(plots)

    # If layout is NULL, then use 'cols' to determine layout
    if (is.null(layout)) {
        # Make the panel
        # ncol: Number of columns of plots
        # nrow: Number of rows needed, calculated from # of cols
        layout <- matrix(seq(1, cols * ceiling(numPlots / cols)),
        ncol = cols,
        nrow = ceiling(numPlots / cols))
    }

    if (numPlots == 1) {
        print(plots[[1]])
    } else {
        # Set up the page
        grid.newpage()
        pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))

        # Make each plot, in the correct location
        for (i in 1 : numPlots) {
            # Get the i,j matrix positions of the regions that contain this subplot
            matchidx <-
            as.data.frame(which(layout == i, arr.ind = TRUE))

            print(plots[[i]],
            vp = viewport(
            layout.pos.row = matchidx$row,
            layout.pos.col = matchidx$col
            ))
        }
    }
}



read_nc_file = function(file_path, name1) {
    library(ncdf4)
    nc <- nc_open(file_path)

    var1 <- ncvar_get(nc, name1)


    result = list(values = var1)
    nc_close(nc)
    rm(nc)
    return(result)
}


if (TRUE)
{
    res.mat.u = array(dim = c(120, 125, 1461))
    res.mat.v = array(dim = c(120, 125, 1461))



    con <-
    file("wind-noise-exp\\wind-noise\\uv.era.2014.txt", "r", blocking = FALSE)
    fil.lines = readLines(con)
    close(con)


    print("START READ")
    #for (i in 1:1461)
    for (i in 1 : 1300)
    {
        uv.step.raw = fil.lines[(2 + 251 * (i - 1)) : ((1 + 251 * (i - 1)) + 250)]
        uv.step.tmp = sapply(uv.step.raw, function(x) {
            strsplit(x, " ")
        })

        uv.step = (t(simplify2array(uv.step.tmp)))

        class(uv.step) <- "numeric"



        u.vals = t(uv.step[1 : 125,])
        v.vals = t(uv.step[126 : 250,])


        res.mat.u[, , i] = u.vals
        res.mat.v[, , i] = v.vals
    }
    print("END READ")
} else {
    res.mat.u = readRDS("wind-noise-exp//res.mat.u_v_saved.rds")
    res.mat.v = readRDS("wind-noise-exp//res.mat.v_v_saved.rds")
}

if (TRUE) {
    cor.local.u.mat = array(dim = c(120, 125, 120, 125))
    cor.local.v.mat = array(dim = c(120, 125, 120, 125))

    print ("START CALC CORS")

    for (i in 1 : dim(u.vals)[1]) {
        print(i)
        for (j in 1 : dim(u.vals)[2]) {
            cor.local.u.mat[i, j, ,] = apply(res.mat.u, 1 : 2, cor, res.mat.u[i, j,])
            cor.local.v.mat[i, j, ,] = apply(res.mat.v, 1 : 2, cor, res.mat.v[i, j,])
        }
    }
} else {
    cor.local.u.mat = readRDS("wind-noise-exp//cor_u_saved.rds")
    cor.local.v.mat = readRDS("wind-noise-exp//cor_v_saved.rds")
}

for (ens_id in 1 : 30)
{
    print(glue("ENS ID {ens_id}"))
    res.mat.u.noised = array(dim = c(120, 125, 1461))
    res.mat.v.noised = array(dim = c(120, 125, 1461))


    #for (time.step.id in 1:1461)
    for (time.step.id in 1 : 1300)
    {
        print(glue("TS-{time.step.id}"))
        u.vals = res.mat.u[, , time.step.id]
        v.vals = res.mat.v[, , time.step.id]

        cor.uv.mat = res.mat.u[, , 1]

        print("START ANALYSE U-V COR")
        if (FALSE)
        {
            for (i in 1 : dim(u.vals)[1]) {
                for (j in 1 : dim(u.vals)[2]) {
                    cor.uv.mat[i, j] = cor(res.mat.u[i, j,], res.mat.v[i, j,])
                }
            }
        }
        print("END ANALYSE U-V COR")

        #noise

        N = round(runif(1, 5, 25))
        dt.noise.point = data.table(
        id = 1 : N,
        n.u = 1.0,
        n.v = 1.0,
        x1 = 1 : N,
        y1 = 1 : N
        )

        #create noise points
        if (time.step.id != 1)
        {
            total.noise.matrix.u.prev = total.noise.matrix.u
            total.noise.matrix.v.prev = total.noise.matrix.v
        }

        total.noise.matrix.u = u.vals * 0
        total.noise.matrix.v = v.vals * 0




        print("START CREATE NOISE POINTS")

        for (np_ind in 1 : N)
        {
            i.p = round(runif(1, 1, dim(u.vals)[1]))
            j.p = round(runif(1, 1, dim(u.vals)[2]))

            real.val.u = u.vals[i.p, j.p]

            cor.local.u = cor.local.u.mat[i.p, j.p, ,]

            cor.local.v = cor.local.v.mat[i.p, j.p, ,]

            sigma.noise.rel = 0.25

            f.noise.u = rnorm(1, 0, abs(real.val.u) * sigma.noise.rel)


            cor.vu = cor.uv.mat[i.p, j.p]

            real.val.v = v.vals[i.p, j.p]
            f.noise.v = rnorm(1, 0, abs(real.val.v) * sigma.noise.rel) * (1 - cor.vu) + f.noise.u * cor.vu

            dt.noise.point[np_ind,]$n.u = f.noise.u
            dt.noise.point[np_ind,]$n.v = f.noise.v

            dt.noise.point[np_ind,]$x1 = i.p
            dt.noise.point[np_ind,]$y1 = j.p

            total.noise.matrix.u = total.noise.matrix.u + f.noise.u * (cor.local.u)
            total.noise.matrix.v = total.noise.matrix.v + f.noise.v * (cor.local.v)
        }

        print("END CREATE NOISE POINTS")

        cor.prev = 0.

        #add noise to field
        if (time.step.id != 1)
        {
            u.vals.noised = u.vals + ((total.noise.matrix.u * (1 - cor.prev) + total.noise.matrix.u.prev *
            cor.prev))
            v.vals.noised = v.vals + (total.noise.matrix.v * (1 - cor.prev) + total.noise.matrix.v.prev *
            cor.prev)
        } else {
            u.vals.noised = u.vals + total.noise.matrix.u
            cor.prev
            v.vals.noised = v.vals + total.noise.matrix.v
        }

        res.mat.u.noised[, , time.step.id] = u.vals.noised
        res.mat.v.noised[, , time.step.id] = v.vals.noised


        points.num = dim(u.vals)[1] * dim(u.vals)[2]


        wind.dt = data.frame(
        x = 1 : points.num,
        y = 1 : points.num,
        spd_orig = 1 : points.num,
        u_orig = 1 : points.num,
        v_orig = 1 : points.num,
        spd_noised = 1 : points.num,
        u_noised = 1 : points.num,
        v_noised = 1 : points.num
        )

        dt.ind = 1


        print("START PREPARE VIS")

        if (FALSE)
        {
            for (i in 1 : dim(u.vals)[1]) {
                for (j in 1 : dim(u.vals)[2]) {
                    wind.dt[dt.ind,]$x = i
                    wind.dt[dt.ind,]$y = j
                    wind.dt[dt.ind,]$u_orig = u.vals[i, j]
                    wind.dt[dt.ind,]$v_orig = v.vals[i, j]

                    wind.dt[dt.ind,]$spd_orig = sqrt(u.vals[i, j] ^ 2 + v.vals[i, j] ^ 2)

                    wind.dt[dt.ind,]$u_noised = u.vals.noised[i, j]
                    wind.dt[dt.ind,]$v_noised = v.vals.noised[i, j]

                    wind.dt[dt.ind,]$spd_noised = sqrt(wind.dt[dt.ind,]$u_noised ^ 2 + wind.dt[dt.ind,]$v_noised ^
                    2)

                    dt.ind = dt.ind + 1
                }
            }

            print("END PREPARE VIS")



            library(ggplot2)
            library(dplyr)
            library('RColorBrewer')


            wind.arrows <- wind.dt %>%
            filter(x %in% sort(unique(x))[c(TRUE, FALSE, FALSE, FALSE, FALSE)],
            y %in% sort(unique(y))[c(TRUE, FALSE, FALSE, FALSE, FALSE)])


            base_plt = ggplot(wind.dt,
            aes(x = x ,
            y = y,
            fill = spd_orig,)) +
                geom_raster() +
                scale_fill_distiller(palette = "Spectral", limits = c(0, 15)) +
                geom_segment(
                data = wind.arrows,
                aes(
                xend = x + u_orig / 0.7,
                yend = y + v_orig / 0.7
                ),
                arrow = arrow(length = unit(0.15, "cm")),
                size = 0.7
                ) +
                coord_equal(expand = 0) +
                theme(
                legend.position = 'bottom',
                legend.direction = 'horizontal',
                panel.background = element_rect(fill = 'white', colour = 'white')
                )


            wind.arrows <- wind.dt %>%
            filter(x %in% sort(unique(x))[c(TRUE, FALSE, FALSE, FALSE, FALSE)],
            y %in% sort(unique(y))[c(TRUE, FALSE, FALSE, FALSE, FALSE)])


            new_plt = ggplot() +
                geom_raster(data = wind.dt,
                aes(x = x ,
                y = y,
                fill = spd_noised)) +
                scale_fill_distiller(palette = "Spectral", limits = c(0, 15)) +
                geom_segment(
                data = wind.arrows,
                aes(
                x = x,
                y = y,
                xend = x + u_noised / 0.7,
                yend = y + v_noised / 0.7
                ),
                arrow = arrow(length = unit(0.15, "cm")),
                size = 0.7
                ) +
                geom_point(
                data = dt.noise.point,
                aes(x = x1, y = y1),
                colour = "blue",
                size = 3
                ) +
                coord_equal(expand = 0) +
                theme(
                legend.position = 'bottom',
                legend.direction = 'horizontal',
                panel.background = element_rect(fill = 'white', colour = 'white')
                )

            ggsave(
            glue(
            "D:\\Work\\wind-noise\\out\\base\\base-exp-ts-{time.step.id}.png"
            ),
            plot = base_plt
            )

            ggsave(
            glue(
            "D:\\Work\\wind-noise\\out\\mod1\\mod-exp-ts-{time.step.id}.png"
            ),
            plot = new_plt
            )
        }
    }



    #########

    if (TRUE)
    {
        library(ncdf4)
        library(lubridate)
        library(data.table)
        library(abind)
        library(data.table)
        library(units)

        gc()

        options(max.print = .Machine$integer.max)

        Sys.setenv(TZ = 'UTC')
        getNcTime <- function(nc) {
            ##NEW VERSION, with the units package
            require(units)
            require(ncdf4)
            options(warn = 1) #show warnings by default
            if (is.character(nc))
            nc <- nc_open(nc)
            ncdims <- names(nc$dim) #get netcdf dimensions
            timevar <-
            ncdims[which(ncdims %in% c(
            "time",
            "Time",
            "datetime",
            "Datetime",
            "date",
            "Date",
            "XTIME"
            ))] #find (first) time variable
            if (length(timevar) > 1) {
                warning(paste("Found more than one time var. Using the first:", timevar[1]))
                timevar <- timevar[1]
            }
            if (length(timevar) != 1)
            stop("ERROR! Could not identify the correct time variable")
            times <- ncvar_get(nc, timevar) #get time data
            timeatt <- ncatt_get(nc, timevar) #get attributes
            timeunit <- timeatt$units
            units(times) <- make_unit(timeunit)
            as.POSIXct(times)
        }
        read_nc_file2 = function(file_path, name1, name2) {
            library(ncdf4)
            nc <- nc_open(file_path)

            var1 <- ncvar_get(nc, name1)
            var2 <- ncvar_get(nc, name2)

            date = getNcTime(nc)

            result = list(values1 = var1,
            values2 = var2,
            date = date)
            nc_close(nc)
            rm(nc)
            return(result)
        }

        gc()

        yearSets = list(c(2014))
        #yearSetsLabels=c("2016","2017")
        #for (yearSetId in 1:2)
        yearSetId = 1
        {
            fileYearSet = yearSets[[yearSetId]]
            fl = glue("wind-noise-exp//wind-noise2//era-uv-2014-ns{ens_id}.txt")
            isFirstYear = TRUE


            for (year in fileYearSet)
            {
                gc()
                print(year)

                u_list_full = res.mat.u.noised
                v_list_full = res.mat.v.noised

                real_dates = (seq(
                from = as.POSIXct(paste(year, "-01-01 00:00:00", sep = ""), tz = "UTC"),
                to = as.POSIXct(paste(year + 1, "-01-01 00:00:00", sep = ""), tz =
                "UTC"),
                by = "6 hour"
                ))


                file_dates = (real_dates)

                dates_found = file_dates[file_dates %in% real_dates]
                dates_not_found_len = length(real_dates) - length(dates_found)

                subs = which(file_dates %in% real_dates)

                u_list = u_list_full[, , subs]
                v_list = v_list_full[, , subs]

                if (dates_not_found_len > 0)
                {
                    add_subs = subs[(length(subs) - dates_not_found_len) : length(subs)]
                    u_list = abind(u_list, u_list_full[, , add_subs], along = 3)
                    v_list = abind(v_list, v_list_full[, , add_subs], along = 3)
                }

                year_size = length(real_dates)



                library(stringr)
                form_dates = gsub(":00:00", "", real_dates)
                form_dates = gsub("-", "", form_dates)
                form_dates = paste(form_dates, "0000", sep = "")



                for (i in 1 : year_size)
                {
                    if (TRUE)
                    {
                        if (year_size != 1)
                        {
                            form_uv = rbind(t(u_list[, , i]), t(v_list[, , i]))
                        } else {
                            form_uv = rbind(t(u_list), t(v_list))
                        }

                        form_uv_txt = as.data.table((round((
                        form_uv
                        ), 3)))
                    }


                    print(form_dates[i])
                    write(form_dates[i], fl, append = ! ((year == fileYearSet[1]) &
                    (i == 1)))

                    fwrite(
                    form_uv_txt,
                    fl,
                    append = TRUE,
                    sep = " ",
                    row.names = FALSE,
                    col.names = FALSE,
                    quote = FALSE,
                    nThread = 32
                    )
                }
            }
        }
    }
}