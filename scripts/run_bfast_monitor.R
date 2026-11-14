# External GPL bfast dependency; no author GEE source is copied or modified.
# Fixed annual monitoring periods are an explicit wrapper, not native recurrence.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args) >= 2)
.libPaths(c(normalizePath(args[1]), .libPaths()))
stopifnot(as.character(packageVersion("bfast")) == "1.7.2")
if (args[2] == "inspect") {
  print(bfast::bfastmonitor)
  print(getFromNamespace(".bfastmonitor.matrix","bfast"))
  quit(status=0)
}
input <- jsonlite::fromJSON(args[2], simplifyVector=FALSE)
out <- args[3]
compact <- length(args) >= 4 && identical(args[4], "compact")
dir.create(out, recursive=TRUE, showWarnings=FALSE)
rows <- list()
for (job in input) {
  x <- vapply(job$values, function(v) if(is.null(v)) NA_real_ else as.numeric(v), numeric(1))
  series <- ts(x, start=c(job$start_year,1), frequency=12)
  warnings <- character()
  fit <- tryCatch(withCallingHandlers(
    bfast::bfastmonitor(series, start=c(job$monitor_year,1),
      formula=response ~ harmon, order=as.integer(job$order), history="all",
      type="OLS-MOSUM", h=job$h, level=job$level, plot=FALSE),
    warning=function(w) { warnings <<- c(warnings,conditionMessage(w)); invokeRestart("muffleWarning") }),
    error=function(e) e)
  row <- list(trajectory_index=job$trajectory_index, monitor_year=job$monitor_year,
              parameter_id=job$parameter_id, warning=paste(unique(warnings),collapse=" | "))
  if (inherits(fit,"error")) {
    row$status <- "failed"; row$error <- conditionMessage(fit)
  } else {
    row$status <- "success"
    row$breakpoint <- as.numeric(fit$breakpoint)
    row$magnitude_native <- as.numeric(fit$magnitude)
    # Full historical runs retain every native fit. Pixel-scale production can
    # request compact mode: the alarm statistics are unchanged, while the
    # hundreds of thousands of fitted R objects are not materialized.
    if (!compact) {
      saveRDS(fit, file.path(out,paste0(job$parameter_id,"_",job$trajectory_index,"_",job$monitor_year,".rds")))
    }
    trace <- as.data.frame(fit$tspp)
    # The matrix implementation leaves the fitted-value column named V3.
    stopifnot(ncol(trace) >= 3, colnames(trace)[2] == "time")
    colnames(trace)[1:3] <- c("response","time","prediction")
    if (!compact) {
      row$trace <- trace
      row$coefficients <- as.list(coef(fit$model))
    }
    if (!is.na(fit$breakpoint)) {
      at <- which.min(abs(trace$time-fit$breakpoint))
      width <- floor(job$h * sum(trace$time < job$monitor_year))
      idx <- seq.int(max(1,at-width+1),at)
      row$alarm_residual_mean <- mean(trace$response[idx]-trace$prediction[idx])
    }
  }
  rows[[length(rows)+1]] <- row
  if (length(rows) %% 100 == 0) {cat(length(rows),"/",length(input)," BFAST jobs\n"); flush.console()}
}
jsonlite::write_json(rows,file.path(out,"results.json"),auto_unbox=TRUE,na="null",digits=16)
capture.output(sessionInfo(),file=file.path(out,"session_info.txt"))
