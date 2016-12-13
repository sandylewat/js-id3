function log2(n) {
    return Math.log(n)/Math.log(2);
}

function I(data, classAttribute) {
    var n = data.length;
    var sum = 0;
    classAttribute.possibleValues().forEach(function(a) {
        var nA = data.filter(function(d){
            return d.fields()[classAttribute.name()] === a;
        }).length;
        var pA = nA/n;
        if(nA != 0)
        {
            sum += pA*log2(pA);    
        }
    });
    return sum*-1;
}

function Info(attribute, data, classAttribute) {
    var n = data.length;
    var sum = 0;
    attribute.possibleValues().forEach(function(value) {
        var nA = data.filter(function(d) {
            return d.fields()[attribute.name()] === value;
        }).length;
        var dataA = data.filter(function(d){
            return d.fields()[attribute.name()] === value;
        });
        var Ia=I(dataA,classAttribute);
        sum += nA/n*Ia;
    });
    return sum;
}
