# Q Learning
Repositori ini berisikan environment game yang dipakai untuk salah satu algoritma reinforcement learning yaitu q learning.

https://user-images.githubusercontent.com/56197074/126945647-71acc50c-3463-4150-b35e-059ad06fcbbb.mp4

## How To Use
Berikut merupakan tatacara menggunakan source code q learning ini:

1. Pastikan Python versi 3.7 ke atas tersedia (program ini dibuat di Python 3.8.5)
2. Pastikan library numpy dan IPython.display tersedia
3. Siapkan file ipynb untuk mendapat pengalaman melihat agen melakukan training secara maksimal
4. Lakukan import source code

```python
import qlearning
```

5. Siapkan game environment

```python
env = qlearning.BananaHole()
```

6. Train agen menggunakan fungsi q_learning

```python
q_table,win,lose = qlearning.q_learning(env=env,n_eps=1000,st_per_eps=100,learning_rate=0.1,disc_rate=0.99,explore_decay_rate=0.3,render=True)
```
```
Penjelasan mengenai fungsi q_learning:
env : Game environment
n_eps : int (jumlah episode)
st_per_eps : int (jumlah langkah per episode)
learning_rate : float, 0-1 (kecepatan pembelajaran agen)
disc_rate : float, 0-1 (nilai gamma)
explore_decay_rate : float, 0-1 (laju decay dari masa eksplorasi)
render : boolean (jika ingin melihat secara langsung pembelajaran yang dilakukan oleh agen)
```
