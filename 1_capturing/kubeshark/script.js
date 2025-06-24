var xxx = [];
function onItemCaptured(metadata) {
  xxx.push(metadata);
}
console.log("!! start_capture !!");

function writeToFile() {
  // print current tiemstamp date
  console.log(new Date().toISOString());
  console.log("!! writeToFile called after 60 secs !!");
  file.mkdir("ks");
  var tempFile = file.temp("shark", "ks", "json");
  if (xxx.length > 0) {
    var yyy = JSON.stringify(xxx);
    file.write(tempFile, yyy);
    console.log("Written to " + tempFile + " length: " + xxx.length)
    xxx.length = 0;
  }
}

jobs.schedule("write-to-file", "*/60 * * * * *", writeToFile);
