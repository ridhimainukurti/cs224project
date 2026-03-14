//modified for each Year and Landsat Data Type 
var trainingExtent1990 = ee.FeatureCollection([
  ee.Feature(ee.FeatureCollection(urban_1990).geometry()),
  ee.Feature(ee.FeatureCollection(veg_1990).geometry()),
  ee.Feature(ee.FeatureCollection(farm_1990).geometry()),
  ee.Feature(ee.FeatureCollection(bare_1990).geometry()),
  ee.Feature(ee.FeatureCollection(water_1990).geometry())
]).geometry();

// Add a modest buffer around all training polygons
var riverside = trainingExtent1990.buffer(5000).bounds();

function applyScaleFactors(image) {
  var optical = image.select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4'])
                     .multiply(2.75e-05)
                     .add(-0.2);
  return image.addBands(optical, null, true);
}

function maskLandsatClouds(image) {
  var qa = image.select('QA_PIXEL');
  var cloudShadowBit = 1 << 4;
  var cloudBit = 1 << 3;

  var mask = qa.bitwiseAnd(cloudShadowBit).eq(0)
               .and(qa.bitwiseAnd(cloudBit).eq(0));

  return image.updateMask(mask);
}

// Build Riverside 1990 composite from Landsat 5
var image1990 = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2')
  .filterBounds(riverside)
  .filterDate('1989-01-01', '1991-12-31')
  .map(applyScaleFactors)
  .map(maskLandsatClouds)
  .map(function(img) {
    return img.select(
      ['SR_B3', 'SR_B2', 'SR_B1', 'SR_B4'],
      ['red', 'green', 'blue', 'nir']
    ).clip(riverside);
  })
  .median()
  .clip(riverside);

// Add NDVI
var image1990_ndvi = image1990.addBands(
  image1990.normalizedDifference(['nir', 'red']).rename('ndvi')
);

// Display image
Map.centerObject(riverside, 10);
Map.addLayer(
  image1990_ndvi,
  {bands: ['red', 'green', 'blue'], min: 0, max: 0.3},
  'Riverside 1990 RGB'
);

// -----------------------------------------------------
// Helper: generate random sample points inside a class polygon
// -----------------------------------------------------
function makeRandomPoints(geomOrFc, nPoints, label, subclass, seed) {
  var geom = ee.FeatureCollection(geomOrFc)
    .geometry()
    .simplify(30);

  var points = ee.FeatureCollection.randomPoints({
    region: geom,
    points: nPoints,
    seed: seed,
    maxError: 30
  });

  return points.map(function(pt) {
    return pt.set({
      label: label,
      subclass: subclass,
      city: 'riverside',
      year: 1990
    });
  });
}

// -----------------------------------------------------
// Choose balanced sample sizes
// -----------------------------------------------------
var urbanPts = makeRandomPoints(urban_1990, 300, 1, 'urban', 1);
var vegPts   = makeRandomPoints(veg_1990,   100, 0, 'vegetation', 2);
var farmPts  = makeRandomPoints(farm_1990,  100, 0, 'farmland', 3);
var barePts  = makeRandomPoints(bare_1990,  100, 0, 'bare_soil', 4);
var waterPts = makeRandomPoints(water_1990, 100, 0, 'water', 5);

// Merge all labeled points
var trainingPoints1990 = urbanPts
  .merge(vegPts)
  .merge(farmPts)
  .merge(barePts)
  .merge(waterPts);

// Optional: show points on map
Map.addLayer(trainingPoints1990, {}, 'Training Points 1990', false);

// -----------------------------------------------------
// Sample image values only at those random points
// -----------------------------------------------------
var samples1990 = image1990_ndvi.sampleRegions({
  collection: trainingPoints1990,
  properties: ['label', 'subclass', 'city', 'year'],
  scale: 30,
  geometries: false,
  tileScale: 4
});

// -----------------------------------------------------
// Quick checks
// -----------------------------------------------------
print('Total 1990 samples', samples1990.size());
print('Preview rows', samples1990.limit(10));
print('Counts by label', samples1990.aggregate_histogram('label'));
print('Counts by subclass', samples1990.aggregate_histogram('subclass'));

// -----------------------------------------------------
// Export to Drive
// -----------------------------------------------------
Export.table.toDrive({
  collection: samples1990,
  description: 'riverside_1990_training_samples',
  folder: 'urban_expansion_exports',
  fileNamePrefix: 'riverside_1990_training_samples',
  fileFormat: 'CSV',
  selectors: ['red', 'green', 'blue', 'nir', 'ndvi', 'label', 'city', 'year', 'subclass']
});