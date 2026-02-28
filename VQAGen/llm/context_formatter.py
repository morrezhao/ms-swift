"""Scene context formatter for LLM input"""

from typing import Dict, List, Any


class SceneContextFormatter:
    """Converts scene metadata to LLM-readable text context"""

    def format_scene_context(self,
                             scene_name: str,
                             scene_info: Dict[str, Any],
                             frame_info: Dict[str, Any] = None) -> str:
        """
        Format scene metadata as natural language context for LLM.

        Args:
            scene_name: Name of the scene
            scene_info: Scene-level metadata dict
            frame_info: Frame-level metadata dict (optional)

        Returns:
            Formatted context string
        """
        parts = []

        # Basic scene info
        parts.append("=== Scene Information ===")
        parts.append(f"Scene Name: {scene_name}")

        room_size = scene_info.get('room_size')
        if room_size is not None:
            parts.append(f"Room Area: {room_size:.2f} square meters")

        room_center = scene_info.get('room_center')
        if room_center is not None and len(room_center) == 3:
            parts.append(
                f"Room Center: [{room_center[0]:.2f}, {room_center[1]:.2f}, {room_center[2]:.2f}]"
            )

        # Object inventory
        object_counts = scene_info.get('object_counts', {})
        object_bboxes = scene_info.get('object_bboxes', {})

        if object_counts:
            parts.append("\n=== Object Inventory ===")
            parts.append(f"Total categories: {len(object_counts)}")

            for category, count in sorted(object_counts.items()):
                parts.append(f"\n- {category}: {count} instance(s)")

                # Add bbox info if available
                bboxes = object_bboxes.get(category, [])
                for i, bbox in enumerate(bboxes):
                    centroid = bbox.get('centroid', [])
                    axes = bbox.get('axesLengths', [])

                    if centroid and len(centroid) == 3:
                        parts.append(
                            f"  Instance {i+1} position: "
                            f"[{centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f}]"
                        )
                    if axes and len(axes) == 3:
                        parts.append(
                            f"  Dimensions (L/W/H): "
                            f"[{axes[0]:.2f}m, {axes[1]:.2f}m, {axes[2]:.2f}m]"
                        )

        # Summary of single-instance categories (useful for distance questions)
        single_instance_cats = [
            cat for cat, count in object_counts.items() if count == 1
        ]
        if single_instance_cats:
            parts.append("\n=== Single-Instance Categories ===")
            parts.append("(These can be used for distance/size questions)")
            parts.append(", ".join(sorted(single_instance_cats)))

        return "\n".join(parts)

    def format_available_question_types(self) -> str:
        """Return description of available question types"""
        return """
Available Question Types:
1. object_counting - How many objects of a category are in the scene
2. object_size - Estimate the size of an object (in cm)
3. object_abs_distance - Distance between two objects (in meters)
4. room_size - Estimate the room area (in square meters)
"""
