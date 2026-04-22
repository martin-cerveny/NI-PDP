import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def main():
    lines = sys.stdin.read().strip().split('\n')
    grid, price, calls, time = [], "", "", ""

    for line in lines:
        parts = line.strip().split()
        if not parts: continue
        if parts[0].startswith("Calls:"):
            calls = line
        if parts[0].startswith("Cas behu:"):
            time = line
        elif len(parts) == 1 and parts[0].isdigit():
            price = parts[0]
        else:
            grid.append(parts)

    if not grid:
        print("No data to plot.")
        return

    R, C = len(grid), len(grid[0])
    fig, ax = plt.subplots(figsize=(C * 0.8, R * 0.8))
    ax.set_xlim(0, C)
    ax.set_ylim(0, R)
    ax.invert_yaxis()
    ax.axis('off')

    # Draw cells and text
    for r in range(R):
        for c in range(C):
            val = grid[r][c]
            color = "#DDDDDD" # Gray for uncovered cells (numbers)
            if val.startswith("T"): color = "#66B2FF" # Blue for T
            elif val.startswith("Z"): color = "#99FF99" # Green for Z

            # Base cell
            rect = patches.Rectangle((c, r), 1, 1, linewidth=1, edgecolor='lightgray', facecolor=color)
            ax.add_patch(rect)

            # Text (shape ID or price)
            ax.text(c + 0.5, r + 0.5, val, ha='center', va='center', fontsize=12, fontweight='bold')

            # Bold borders between different shapes
            if c < C - 1 and grid[r][c] != grid[r][c+1]:
                ax.plot([c+1, c+1], [r, r+1], color='black', linewidth=3)
            if r < R - 1 and grid[r][c] != grid[r+1][c]:
                ax.plot([c, c+1], [r+1, r+1], color='black', linewidth=3)

    # Outer borders
    ax.plot([0, C, C, 0, 0], [0, 0, R, R, 0], color='black', linewidth=3)

    plt.title(f"Price: {price} | {calls}")
    plt.tight_layout()
    plt.savefig("solution.png", dpi=150)
    print("Visualization saved to solution.png")

if __name__ == "__main__":
    main()