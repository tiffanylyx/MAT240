autowatch = 1;
buffer = new Buffer("foo");

function bang(){
post("got here");
}

function list(a){
  ar = arrayfromargs(a);
  post("got", a);
}
