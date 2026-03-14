//modified for each Year and Landsat Data Type 
var riverside = ee.Geometry.Rectangle([-117.55, 33.85, -117.20, 34.10]);

function applyScaleFactors(image) {
  var optical = image.select('SR_B.*').multiply(2.75e-05).add(-0.2);
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

var image1990 = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2')
  .filterBounds(riverside)
  .filterDate('1989-01-01', '1991-12-31')
  .map(applyScaleFactors)
  .map(maskLandsatClouds)
  .median()
  .select(['SR_B3', 'SR_B2', 'SR_B1'], ['red', 'green', 'blue']);

Map.centerObject(riverside, 10);

Map.addLayer(
  image1990,
  {bands: ['red', 'green', 'blue'], min: 0, max: 0.3},
  'Riverside 1990 RGB'
);