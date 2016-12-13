function DecisionTree($, ko, settings) {
    var self = this;

    function TupleModel(tree = null) {
        var self = this;
        self.fields = ko.observableArray();
        self.isTrainingData = ko.observable(true);
        self.tree = ko.observable(tree);
        self.prediction = ko.computed(function(){
            if (self.tree() != null) {
                var node = self.tree();
                while (!node.isClass()) {
                    node = node.getNextNode(self.fields());
                    if (node == null) {
                        return null;
                    }
                }
                return node.classValue();
            }
            return null;
        }); 
    }

    function NodeModel(attribute, value, id) {
        var self = this;

        self.isFailure = ko.observable(false);
        self.attribute = ko.observable(attribute);
        self.value = ko.observable(value);
        self.children = ko.observableArray();
        self.isClass = ko.observable(false);
        self.classValue = ko.observable();
        self.id = ko.observable(id);

        self.getNextNode = function (fields) {
            if (self.isClass()) {
                return self.classValue;
            }
            for(var i=0; i< self.children().length; ++i) {
                var child =  self.children()[i];
                if (fields[self.attribute().name()] === child.value()) {
                    return child;
                }
            }
            return null;
        };
    }

    function EntropyModel(attributeName, value, gain) {
        var self = this;

        self.attributeName = ko.observable(attributeName);
        self.value = ko.observable(value);
        self.gain = ko.observable(gain);

    }

    function StepModel() {
        var self = this;

        self.infoD = ko.observable();
        self.entropies = ko.observableArray();
        self.maxGainAttribute = ko.observable();
        self.homogen = ko.observable(false);
        self.classification = ko.observable();
        self.value = ko.observable();
        self.id = ko.observable();
    }

    function ID3Model(attributes, data) {
        var self = this;

        self.tree = ko.observable();
        self.trainingData = ko.observableArray(data.filter(function (d) {
            return d.isTrainingData();
        }));
        self.testData = ko.observableArray(data.filter(function (d) {
            return !d.isTrainingData();
        }));
        self.attributes = ko.observableArray(attributes.filter(function (a) {
            return !a.isClass();
        }));
        self.class = ko.observable(attributes.filter(function (a) {
            return a.isClass();
        })[0]);
        self.buildSteps = ko.observableArray([]);      

        self.build = function () {
            self.tree(self.id3(self.attributes(), self.class(), self.trainingData(), "root"));
            self.testData().forEach(function (d) {
                d.tree(self.tree());
            });
        }

        self.removeTestRow = function (row) {
            self.testData.remove(row);
        }

        self.addTestRow = function () {
            self.testData.push(new TupleModel(self.tree()));
        }

        self.id = ko.observable(0);

        self.id3 = function (_attributes, _class, _data, label) {
            if (_data.length == 0) {
                self.id(self.id() + 1);
                var fail = new NodeModel(undefined, label, self.id());
                fail.isFailure(true);
                return fail;

            }

            var sumCV = [];
            _class.possibleValues().forEach(function (v) {
                sumCV.push({
                    name: v,
                    sum: _data.filter(function (d) {
                        return d.fields()[_class.name()] == v;
                    }).length
                })
            });
            var mostCommon = sumCV.sort(function (a, b) {
                return b.sum - a.sum;
            })[0];
            if (mostCommon.sum == _data.length) {
                self.id(self.id() + 1);
                var node = new NodeModel(_class, label, self.id());
                node.isClass(true);
                node.classValue(mostCommon.name);
                var step = new StepModel();
                step.infoD(I(_data, _class));
                step.homogen(true);
                step.classification(mostCommon.name);
                step.value(label);
                step.id(self.id())
                self.buildSteps.push(step);
                return node;
            }
            if (_attributes.length == 0) {
                self.id(self.id() + 1);
                var node = new NodeModel(_class, label, self.id());
                node.isClass(true);
                node.classValue(mostCommon.name);
                return node;
            }

            var step = new StepModel();
            var infoD = I(_data, _class);
            step.infoD(infoD);

            var infoA = [];
            _attributes.forEach(function (a) {
                var entropy = Info(a, _data, _class);
                var gain = infoD - entropy;
                infoA.push({
                    name: a.name(),
                    value: gain
                });
                step.entropies.push(new EntropyModel(a.name(), entropy, gain));
            });

            var maxGainAttribute = infoA.sort(function (a, b) {
                return b.value - a.value;
            })[0];
            maxGainAttribute = _attributes.filter(function (d) {
                return d.name() == maxGainAttribute.name;
            })[0];
            step.maxGainAttribute(maxGainAttribute.name());

            self.buildSteps.push(step);
            var _newAttributes = _attributes.filter(function (a) {
                return a.name() != maxGainAttribute.name();
            })
            self.id(self.id() + 1);
            step.id(self.id())
            var node = new NodeModel(maxGainAttribute, label, self.id());
            maxGainAttribute.possibleValues().forEach(function (v) {
                var _newData = _data.filter(function (d) {
                    return d.fields()[maxGainAttribute.name()] == v;
                });
                if (_newData.length > 0) {
                    node.children.push(self.id3(_newAttributes, _class, _newData, v));
                }
            })
            return node;
        }

        self.correctPrediction = ko.computed(function(){
            var sum = 0;
            for(var i=0; i<self.testData().length; ++i) {
                var data = self.testData()[i];
                if(data.prediction() == data.fields()[self.class().name()]) {
                    ++sum;
                }
            }
            return sum;
        });
        self.wrongPrediction = ko.computed(function(){
            if(self.testData().length > 0) {
                return self.testData().length - self.correctPrediction();
            }
            return 0;   
        });
        self.accuracy = ko.computed(function(){
            if(self.testData().length > 0) {
                return self.correctPrediction()/self.testData().length;
            }
            return 0;   
        });

    }

    function AttributeModel() {
        var self = this;

        self.name = ko.observable("");
        self.possibleValuesString = ko.observable("");
        //ID3 will only handle nominal data type
        //we can implement C4.5 or J48 with this model later
        self.dataType = ko.observable("nominal");
        self.possibleValues = ko.computed(function () {
            return self.possibleValuesString().split(",").map(Function.prototype.call, String.prototype.trim);
        });

        self.isClass = ko.observable(false);

        self.isNameError = ko.computed(function () {
            return self.name().length <= 0;
        });

        self.arePossibleValuesError = ko.computed(function () {
            return self.possibleValuesString() == "";
        });

        self.isValid = ko.computed(function () {
            return !self.isNameError() && !self.arePossibleValuesError();
        });

    }

    self.init = function (id) {
        self.viewModel = new ViewModel();
        var selectedElement = document.getElementById(id);
        ko.applyBindings(self.viewModel, selectedElement);
    }

    function ViewModel() {
        var self = this;
        self.state = ko.observable(1);
        self.relationName = ko.observable("");
        self.attributes = ko.observableArray([new AttributeModel()]);
        self.data = ko.observableArray();
        self.class = ko.observable();
        self.rawData = ko.observable();
        self.id3 = ko.observable();

        self.isRelationNameError = ko.computed(function () {
            return self.relationName().length < 1;
        });

        self.areAttributesValid = ko.computed(function () {
            if (self.attributes().length < 1) {
                return false;
            }
            for (var i = 0; i < self.attributes().length; ++i) {
                if (!self.attributes()[i].isValid()) {
                    return false;
                }
            }

            return true;
        });

        self.isClassError = ko.computed(function () {
            return typeof self.class() === "undefined";
        });


        self.goToNextState = function () {
            if (self.state() > 0 && self.state() < 4)
                self.state(self.state() + 1);
            return self.state();
        }

        self.goToPreviousState = function () {
            if (self.state() > 1 && self.state() < 4)
                self.state(self.state() - 1);
            return self.state();
        }

        self.addAtribute = function () {
            self.attributes.push(new AttributeModel());
        }

        self.removeAttribute = function (attribute) {
            self.attributes.remove(attribute);
        }

        self.addRow = function () {
            self.data.push(new TupleModel());
        }

        self.removeRow = function (row) {
            self.data.remove(row);
        }

        self.setClass = function (attribute) {
            if (attribute.isClass() == true) {
                var prev = self.class();
                if (typeof prev !== "undefined") {
                    prev.isClass(false);
                }

                self.class(attribute);
            }
            else {
                self.class(undefined);
            }

            return true;
        }

        self.isReadyToBuild = ko.computed(function(){
            return self.data().length > 0 && typeof self.class() != 'undefined';
        });
        

        self.fileLoadedCallback = function (file, data) {
            self.rawData(data);
        }

        self.createEmpty = function () {
            self.data([]);
            self.attributes([]);
            self.relationName("");
            self.goToNextState();
        }

        self.parseArff = function () {
            self.data([]);
            self.attributes([]);
            self.relationName("");

            var arffString = self.rawData();
            arffArray = arffString.split("\r\n");
            var readData = false;
            for (var i = 0; i < arffArray.length; ++i) {
                var line = arffArray[i];
                if (line.length > 0) {
                    if (!readData) {
                        var token = line.split(" ");
                        if (token.length > 0) {
                            var key = token[0];
                            if (key === "@relation" && token.length > 1) {
                                self.relationName(token[1]);
                            }
                            else if (key === "@attribute" && token.length > 2) {
                                attribute = new AttributeModel();
                                attribute.name(token[1]);
                                values = line.substr(line.indexOf("{")).replace("{", "").replace("}", "");
                                attribute.possibleValuesString(values);
                                self.attributes.push(attribute);
                            }
                            else if (key === "@data") {
                                readData = true;
                            }
                        }
                    }
                    else {
                        var token = line.split(",");
                        if (token.length == self.attributes().length) {
                            var tuple = new TupleModel();
                            var finished = false;
                            for (var j = 0; j < token.length; ++j) {
                                var tok = token[j].trim();
                                if (self.attributes()[j].possibleValues().indexOf(tok) > -1) {
                                    tuple.fields()[self.attributes()[j].name()] = tok;
                                    if (j == token.length - 1) {
                                        finished = true;
                                    }
                                }
                            }
                            if (finished) {
                                self.data.push(tuple);
                            }
                        }
                    }
                }
            }
            self.goToNextState();
        }

        self.buildID3 = function () {
            self.id3(new ID3Model(self.attributes(), self.data()));
            self.id3().build();
            self.goToNextState();
            drawTree(self.id3().tree());
        }
    }
}

