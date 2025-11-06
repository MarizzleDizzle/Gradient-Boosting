"""
Tree structure visualizer for individual decision trees
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np


class TreeVisualizer:
    """
    Visualizer for individual decision tree structures.
    """

    def __init__(self, tree):
        """
        Initialize the tree visualizer.

        Parameters:
        -----------
        tree : DecisionTreeRegressor
            A decision tree from the gradient boosting model
        """
        self.tree = tree

    def plot_tree(self, feature_names=None, figsize=(20, 12)):
        """
        Visualize the tree structure.

        Parameters:
        -----------
        feature_names : list or None
            Names of features
        figsize : tuple
            Figure size (width, height)
        """
        if self.tree.root is None:
            print("Tree has not been fitted yet!")
            return

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # Calculate tree layout
        positions = {}
        self._calculate_positions(self.tree.root, 0.5, 0.95, 0.25, positions)

        # Draw tree
        self._draw_tree(ax, self.tree.root, positions, feature_names)

        plt.title('Decision Tree Structure', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()

    def _calculate_positions(self, node, x, y, width, positions, node_id=0):
        """
        Calculate positions for tree nodes.

        Parameters:
        -----------
        node : TreeNode
            Current tree node
        x : float
            X position
        y : float
            Y position
        width : float
            Width for child nodes
        positions : dict
            Dictionary to store node positions
        node_id : int
            Unique identifier for node

        Returns:
        --------
        int : Updated node_id
        """
        positions[node_id] = (x, y)

        if node.value is None:  # Not a leaf node
            # Left child
            left_id = node_id + 1
            node_id = self._calculate_positions(node.left, x - width, y - 0.15,
                                               width * 0.5, positions, left_id)

            # Right child
            right_id = node_id + 1
            node_id = self._calculate_positions(node.right, x + width, y - 0.15,
                                               width * 0.5, positions, right_id)

        return node_id

    def _draw_tree(self, ax, node, positions, feature_names, node_id=0):
        """
        Recursively draw the tree.

        Parameters:
        -----------
        ax : matplotlib axis
            Axis to draw on
        node : TreeNode
            Current tree node
        positions : dict
            Dictionary of node positions
        feature_names : list or None
            Names of features
        node_id : int
            Unique identifier for node
        """
        x, y = positions[node_id]

        if node.value is not None:  # Leaf node
            # Draw leaf
            box = FancyBboxPatch((x - 0.04, y - 0.03), 0.08, 0.06,
                                boxstyle="round,pad=0.005",
                                edgecolor='green', facecolor='lightgreen',
                                linewidth=2)
            ax.add_patch(box)

            # Add text
            ax.text(x, y, f'Value:\n{node.value:.3f}',
                   ha='center', va='center', fontsize=8, fontweight='bold')
        else:  # Decision node
            # Draw decision node
            box = FancyBboxPatch((x - 0.05, y - 0.03), 0.10, 0.06,
                                boxstyle="round,pad=0.005",
                                edgecolor='blue', facecolor='lightblue',
                                linewidth=2)
            ax.add_patch(box)

            # Add text
            feature_name = (feature_names[node.feature] if feature_names
                          else f'X[{node.feature}]')
            ax.text(x, y, f'{feature_name}\n≤ {node.threshold:.3f}',
                   ha='center', va='center', fontsize=8, fontweight='bold')

            # Draw edges to children
            left_id = node_id + 1
            right_id = self._get_right_child_id(node, node_id)

            x_left, y_left = positions[left_id]
            x_right, y_right = positions[right_id]

            # Left edge (True)
            ax.plot([x, x_left], [y - 0.03, y_left + 0.03], 'k-', linewidth=1.5)
            ax.text((x + x_left) / 2 - 0.02, (y + y_left) / 2, 'True',
                   fontsize=7, style='italic', color='green')

            # Right edge (False)
            ax.plot([x, x_right], [y - 0.03, y_right + 0.03], 'k-', linewidth=1.5)
            ax.text((x + x_right) / 2 + 0.02, (y + y_right) / 2, 'False',
                   fontsize=7, style='italic', color='red')

            # Recursively draw children
            self._draw_tree(ax, node.left, positions, feature_names, left_id)
            self._draw_tree(ax, node.right, positions, feature_names, right_id)

    def _get_right_child_id(self, node, node_id):
        """
        Calculate the node_id for the right child.

        Parameters:
        -----------
        node : TreeNode
            Current tree node
        node_id : int
            Current node id

        Returns:
        --------
        int : Right child node id
        """
        # Count nodes in left subtree
        left_count = self._count_nodes(node.left)
        return node_id + left_count + 1

    def _count_nodes(self, node):
        """
        Count the number of nodes in a subtree.

        Parameters:
        -----------
        node : TreeNode
            Root of subtree

        Returns:
        --------
        int : Number of nodes
        """
        if node is None:
            return 0
        if node.value is not None:
            return 1
        return 1 + self._count_nodes(node.left) + self._count_nodes(node.right)

    def print_tree(self, feature_names=None, indent=""):
        """
        Print the tree structure as text.

        Parameters:
        -----------
        feature_names : list or None
            Names of features
        indent : str
            Indentation string
        """
        if self.tree.root is None:
            print("Tree has not been fitted yet!")
            return

        print("Decision Tree Structure:")
        print("=" * 50)
        self._print_node(self.tree.root, feature_names, indent)

    def _print_node(self, node, feature_names, indent):
        """
        Recursively print tree nodes.

        Parameters:
        -----------
        node : TreeNode
            Current tree node
        feature_names : list or None
            Names of features
        indent : str
            Current indentation
        """
        if node.value is not None:
            print(f"{indent}└─ Leaf: {node.value:.4f}")
        else:
            feature_name = (feature_names[node.feature] if feature_names
                          else f'X[{node.feature}]')
            print(f"{indent}├─ {feature_name} <= {node.threshold:.4f}")
            self._print_node(node.left, feature_names, indent + "│  ")
            print(f"{indent}└─ {feature_name} > {node.threshold:.4f}")
            self._print_node(node.right, feature_names, indent + "   ")

