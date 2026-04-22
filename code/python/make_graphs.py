import matplotlib.pyplot as plt
import numpy as np

# Naměřené časy (T_seq)
t_seq = [505.54, 1387.16, 354.91]

# Vlákna a časy pro OMP
threads_omp = [1, 2, 4, 12, 16, 32, 48]
t_task = [
    [535.41, 321.26, 163.52, 54.71, 35.10, 17.97, 12.38], # Úloha 1
    [1478.52, 750.43, 378.13, 127.50, 94.41, 47.25, 32.09], # Úloha 2
    [376.37, 228.42, 124.00, 32.36, 24.02, 12.04, 8.31]   # Úloha 3
]
t_data = [
    [592.37, 271.83, 133.60, 46.27, 33.59, 20.59, 11.34], # Úloha 1
    [1443.50, 726.53, 365.68, 122.10, 91.42, 45.87, 30.36], # Úloha 2
    [371.44, 184.65, 93.36, 31.24, 23.35, 11.70, 7.97]    # Úloha 3
]

# Vlákna a časy pro MPI (3x48 a 4x48)
threads_mpi = [144, 192]
t_mpi = [
    [5.08, 3.84],   # Úloha 1
    [12.90, 10.05], # Úloha 2
    [4.12, 3.33]    # Úloha 3
]

# Generování grafu pro každou úlohu
for i in range(3):
    uloha = i + 1
    seq = t_seq[i]

    # Výpočet zrychlení (S = T_seq / T_par)
    s_task = [seq / t for t in t_task[i]]
    s_data = [seq / t for t in t_data[i]]
    s_mpi = [seq / t for t in t_mpi[i]]

    # Nastavení plátna
    plt.figure(figsize=(10, 6), dpi=300)

    # Ideální zrychlení
    x_ideal = np.linspace(0, 200, 10)
    plt.plot(x_ideal, x_ideal, '--', color='darkgray', label='Ideální zrychlení')

    # OMP linky
    plt.plot(threads_omp, s_task, marker='o', color='blue', linewidth=1.5, alpha=0.7, label='OMP Task')
    plt.plot(threads_omp, s_data, marker='o', color='red', linewidth=1.5, alpha=0.7, label='OMP Datový')

    # MPI body
    # Místo marker='D' a s=100 použijeme marker='o' (kolečko) a menší velikost s=40
    plt.scatter(threads_mpi, s_mpi, marker='o', color='green', s=40, zorder=5, alpha=0.7, label='MPI Hybrid')

    # Hodnoty u MPI bodů
    for x, y in zip(threads_mpi, s_mpi):
        plt.text(x, y + (max(s_mpi)*0.03), f'{y:.1f}x', color='green', ha='center', va='bottom', fontsize=9)

    # Parametry grafu
    plt.title(f'Zrychlení - Úloha {uloha}')
    plt.xlabel('Celkový počet jader / vláken')
    plt.ylabel(r'Zrychlení ($T_{seq}/T_{par}$)')
    
    # Dynamické limity os podle dat
    plt.xlim(0, 210)
    plt.ylim(0, max(max(s_mpi) * 1.15, 210))
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper left')

    # Uložení na disk
    plt.tight_layout()
    plt.savefig(f'zrychleni_uloha_{uloha}.png', dpi=300)
    plt.close()
