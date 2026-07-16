import utils.model.metamodel as mm

mm.reconstruct(
    slice("2014-10-23", "2022-06-01"),
    450,
    spectral=True,
    offshore_name="WHACS",
    sites=["buoy_Aunuu_Buoy"],
    pca_variance=0.99,
)
