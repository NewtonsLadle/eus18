python3 -m scripts.run machines/jinx2.json generated2/cjweight/coronal/train1.json
python3 -m scripts.prepare_prior machines/jinx2.json generated2/cjweight/coronal/prior1.json
python3 -m scripts.run machines/jinx2.json generated2/cjweight/coronal/train2.json
python3 -m scripts.prepare_prior machines/jinx2.json generated2/cjweight/coronal/prior2.json
python3 -m scripts.run machines/jinx2.json generated2/cjweight/coronal/train3.json
