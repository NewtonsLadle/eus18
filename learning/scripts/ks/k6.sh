python3 -m scripts.run machines/jinx2.json ks/k6/train1.json
python3 -m scripts.prepare_prior machines/jinx2.json ks/k6/prior1.json
python3 -m scripts.run machines/jinx2.json ks/k6/train2.json
python3 -m scripts.prepare_prior machines/jinx2.json ks/k6/prior2.json
python3 -m scripts.run machines/jinx2.json ks/k6/train3.json
