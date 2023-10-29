####################################################################################################
### Skip messages ##################################################################################
####################################################################################################
SKIP_NO_DATASET = "dataset was not provided"
SKIP_BAD_DIMS_3 = "last dim doesn't have 3 channels"
SKIP_NEG_VALUES = "data has negative values"


####################################################################################################
### Batch norm inspectpr messages ##################################################################
####################################################################################################
WARN_NO_BN_SIMPLE = ("The model doesn't have batch norm layers. "
                     "This means that either the layers of the model aren't normalized correctly, "
                     "or that the model was batch norm layers were fused during export (from torch)")
WARN_NO_BN_ADVANCED = ("The model doesn't have batch norm layers. "
                       "Some layers had abnormal distribution.")
ERR_BN_DIST = ("This might indicate the model wasn't trained properly, or that the input isn't "
               "distributed properly")


####################################################################################################
### Clipping inspector messages ####################################################################
####################################################################################################
INFO_CLIP_GENERAL1 = ("Warning is printed if more than {warn_th}% of the range has only {item_th} items. "
                      "Info is printed if more than {info_th}% of the range has only {item_th} items. "
                      "Consider analyzing the data in depth before applying clipping")
INFO_CLIP_GENERAL2 = ("In some cases the range might not be fixable and affected by other factors.")
INFO_CLIP_GENERAL3 = ("In general, activation clipping suggestion if very sensitive to the calibration set. "
                      "Applying activation clipping in some cases might reduce accuracy.")