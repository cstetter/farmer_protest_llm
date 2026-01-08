import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors

browser_market_share = {
    'browsers': ['firefox', 'chrome', 'safari', 'edge', 'ie', 'opera'],
    'market_share': [8.61, 69.55, 8.36, 4.12, 2.76, 2.43],
    'color': ['#5A69AF', '#579E65', '#F9C784', '#FC944A', '#F24C00', '#00B825']
}


class BubbleChart:
    def __init__(self, area, bubble_spacing=0):
        """
        Setup for bubble collapse.

        Parameters
        ----------
        area : array-like
            Area of the bubbles.
        bubble_spacing : float, default: 0
            Minimal spacing between bubbles after collapsing.

        Notes
        -----
        If "area" is sorted, the results might look weird.
        """
        area = np.asarray(area)
        r = np.sqrt(area / np.pi)

        self.bubble_spacing = bubble_spacing
        self.bubbles = np.ones((len(area), 4))
        self.bubbles[:, 2] = r
        self.bubbles[:, 3] = area
        self.maxstep = 2 * self.bubbles[:, 2].max() + self.bubble_spacing
        self.step_dist = self.maxstep / 2

        # calculate initial grid layout for bubbles
        length = np.ceil(np.sqrt(len(self.bubbles)))
        grid = np.arange(length) * self.maxstep
        gx, gy = np.meshgrid(grid, grid)
        self.bubbles[:, 0] = gx.flatten()[:len(self.bubbles)]
        self.bubbles[:, 1] = gy.flatten()[:len(self.bubbles)]

        self.com = self.center_of_mass()

    def center_of_mass(self):
        return np.average(
            self.bubbles[:, :2], axis=0, weights=self.bubbles[:, 3]
        )

    def center_distance(self, bubble, bubbles):
        return np.hypot(bubble[0] - bubbles[:, 0],
                        bubble[1] - bubbles[:, 1])

    def outline_distance(self, bubble, bubbles):
        center_distance = self.center_distance(bubble, bubbles)
        return center_distance - bubble[2] - \
            bubbles[:, 2] - self.bubble_spacing

    def check_collisions(self, bubble, bubbles):
        distance = self.outline_distance(bubble, bubbles)
        return len(distance[distance < 0])

    def collides_with(self, bubble, bubbles):
        distance = self.outline_distance(bubble, bubbles)
        return np.argmin(distance, keepdims=True)

    def collapse(self, n_iterations=50):
        """
        Move bubbles to the center of mass.

        Parameters
        ----------
        n_iterations : int, default: 50
            Number of moves to perform.
        """
        for _i in range(n_iterations):
            moves = 0
            for i in range(len(self.bubbles)):
                rest_bub = np.delete(self.bubbles, i, 0)
                # try to move directly towards the center of mass
                # direction vector from bubble to the center of mass
                dir_vec = self.com - self.bubbles[i, :2]

                # shorten direction vector to have length of 1
                dir_vec = dir_vec / np.sqrt(dir_vec.dot(dir_vec))

                # calculate new bubble position
                new_point = self.bubbles[i, :2] + dir_vec * self.step_dist
                new_bubble = np.append(new_point, self.bubbles[i, 2:4])

                # check whether new bubble collides with other bubbles
                if not self.check_collisions(new_bubble, rest_bub):
                    self.bubbles[i, :] = new_bubble
                    self.com = self.center_of_mass()
                    moves += 1
                else:
                    # try to move around a bubble that you collide with
                    # find colliding bubble
                    for colliding in self.collides_with(new_bubble, rest_bub):
                        # calculate direction vector
                        dir_vec = rest_bub[colliding, :2] - self.bubbles[i, :2]
                        dir_vec = dir_vec / np.sqrt(dir_vec.dot(dir_vec))
                        # calculate orthogonal vector
                        orth = np.array([dir_vec[1], -dir_vec[0]])
                        # test which direction to go
                        new_point1 = (self.bubbles[i, :2] + orth *
                                      self.step_dist)
                        new_point2 = (self.bubbles[i, :2] - orth *
                                      self.step_dist)
                        dist1 = self.center_distance(
                            self.com, np.array([new_point1]))
                        dist2 = self.center_distance(
                            self.com, np.array([new_point2]))
                        new_point = new_point1 if dist1 < dist2 else new_point2
                        new_bubble = np.append(new_point, self.bubbles[i, 2:4])
                        if not self.check_collisions(new_bubble, rest_bub):
                            self.bubbles[i, :] = new_bubble
                            self.com = self.center_of_mass()

            if moves / len(self.bubbles) < 0.1:
                self.step_dist = self.step_dist / 2

    def plot(self, ax, labels, colors=None, min_font=8, max_font=28, 
         cmap_name='viridis', label_font_threshold=10):
        """
        Draw the bubble plot with font size scaled by bubble size.
        Labels are only shown if font size >= label_font_threshold.
        """
        radii = self.bubbles[:, 2]
        r_min, r_max = np.min(radii), np.max(radii)
    
        if colors is None:
            norm = mcolors.Normalize(vmin=r_min, vmax=r_max)
            cmap = cm.get_cmap(cmap_name)
            colors = [cmap(norm(r)) for r in radii]
    
        for i in range(len(self.bubbles)):
            x, y, r = self.bubbles[i, :3]
            circ = plt.Circle((x, y), r, color=colors[i], ec="black", lw=2, alpha=0.7)
            ax.add_patch(circ)
    
            # Scale font size linearly
            font_size = min_font + (r - r_min) / (r_max - r_min) * (max_font - min_font) if r_max > r_min else (min_font + max_font)/2
    
            # Show label only if large enough
            if font_size >= label_font_threshold:
                ax.text(
                    x, y, labels[i],
                    ha='center', va='center',
                    fontsize=font_size,
                    color='white',
                    weight='bold'
                )

    def highlight(self, highlight_indices, ax, labels, colors=None, highlight_color='firebrick', low_alpha=0.2, min_font=8, max_font=28, cmap_name='viridis'):
        """
        Highlight specific bubbles while dimming others.

        Parameters
        ----------
        highlight_indices : list of int
            Indices of bubbles to highlight.
        highlight_color : str or color
            Color for highlighted bubbles.
        low_alpha : float
            Transparency for non-highlighted bubbles.
        """
        radii = self.bubbles[:, 2]
        r_min, r_max = np.min(radii), np.max(radii)

        # Use existing colors or colormap
        if colors is None:
            norm = mcolors.Normalize(vmin=r_min, vmax=r_max)
            cmap = cm.get_cmap(cmap_name)
            colors = [cmap(norm(r)) for r in radii]

        for i in range(len(self.bubbles)):
            x, y, r = self.bubbles[i, :3]

            # Decide bubble color and alpha
            if i in highlight_indices:
                color = highlight_color
                alpha = 0.9
            else:
                color = 'grey' if colors is None else colors[i]
                alpha = low_alpha

            circ = plt.Circle((x, y), r, color=color, ec="black", lw=2, alpha=alpha)
            ax.add_patch(circ)

            # Scale font size linearly
            font_size = min_font + (r - r_min) / (r_max - r_min) * (max_font - min_font) if r_max > r_min else (min_font + max_font)/2
            if i in highlight_indices:
                ax.text(
                    x, y, labels[i],
                    ha='center', va='center',
                    fontsize=font_size,
                    color='white',
                    weight='bold'
                )