import streamlit as st
import plotly.graph_objects as go
from typing import Dict


class Visualizer:
    """Class to create visualizations for prediction results"""
    
    # Color scheme
    COLORS = {
        'elbow': '#ef4444',      # Red
        'finger': '#22c55e',     # Green
        'forearm': '#3b82f6',    # Blue
        'hand': '#eab308',       # Yellow
        'humerus': '#a855f7',    # Purple
        'shoulder': '#ec4899',   # Pink
        'wrist': '#14b8a6',      # Teal
        'fracture': '#ef4444',   # Red for fracture
        'no_fracture': '#22c55e' # Green for no fracture
    }
    
    @staticmethod
    def create_progress_bar(label: str, value: float, color: str, height: int = 60):
        """
        Create a custom progress bar using Plotly
        
        Args:
            label: Label for the progress bar
            value: Value between 0 and 1
            color: Color of the bar
            height: Height of the bar in pixels
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Add the filled bar
        fig.add_trace(go.Bar(
            x=[value * 100],
            y=[label],
            orientation='h',
            marker=dict(
                color=color,
                line=dict(color=color, width=2)
            ),
            text=f'{value*100:.1f}%',
            textposition='inside',
            insidetextanchor='middle',
            textfont=dict(size=14, color='white', family='Arial Black'),
            hoverinfo='none',
            showlegend=False
        ))
        
        # Add the background bar
        fig.add_trace(go.Bar(
            x=[100],
            y=[label],
            orientation='h',
            marker=dict(
                color='rgba(200, 200, 200, 0.3)',
                line=dict(color='rgba(200, 200, 200, 0.5)', width=1)
            ),
            hoverinfo='none',
            showlegend=False
        ))
        
        # Update layout
        fig.update_layout(
            xaxis=dict(
                range=[0, 100], 
                showticklabels=False, 
                showgrid=False,
                zeroline=False
            ),
            yaxis=dict(
                showticklabels=True,
                tickfont=dict(size=12, family='Arial')
            ),
            height=height,
            margin=dict(l=0, r=0, t=5, b=5),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            barmode='overlay'
        )
        
        return fig
    
    @staticmethod
    def display_results(results: Dict):
        """
        Display complete prediction results
        
        Args:
            results: Dictionary containing prediction results
        """
        # Header
        st.markdown("### 🔍 Analysis Results")
        st.markdown("---")
        
        # Bone Type Section
        st.markdown("#### 🦴 Detected Bone Type")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("**Bone Type:**")
            st.markdown("**Confidence:**")
        with col2:
            st.markdown(f"**{results['bone_type'].upper()}**")
            st.markdown(f"**{results['bone_confidence']*100:.1f}%**")
        
        st.markdown("")
        
        # Fracture Detection Section
        st.markdown("#### 🔬 Fracture Analysis")
        
        has_fracture = results['has_fracture']
        fracture_prob = results['fracture_probability']
        
        # Display status with color coding
        if has_fracture:
            st.markdown(
                f"""
                <div style='padding: 20px; background-color: #fee2e2; border-left: 5px solid #ef4444; border-radius: 5px;'>
                    <h3 style='color: #dc2626; margin: 0;'>⚠️ FRACTURE DETECTED</h3>
                    <p style='color: #991b1b; margin: 10px 0 0 0; font-size: 18px;'>
                        Confidence: {fracture_prob*100:.1f}%
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            color = Visualizer.COLORS['fracture']
        else:
            st.markdown(
                f"""
                <div style='padding: 20px; background-color: #dcfce7; border-left: 5px solid #22c55e; border-radius: 5px;'>
                    <h3 style='color: #16a34a; margin: 0;'>✅ NO FRACTURE DETECTED</h3>
                    <p style='color: #166534; margin: 10px 0 0 0; font-size: 18px;'>
                        Confidence: {(1-fracture_prob)*100:.1f}%
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            color = Visualizer.COLORS['no_fracture']
        
        st.markdown("")
        
        # Fracture confidence bar
        fig = Visualizer.create_progress_bar(
            "Fracture Probability",
            fracture_prob,
            color,
            height=70
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # All Bone Type Confidences
        st.markdown("#### 📊 All Bone Type Confidence Scores")
        st.markdown("*Sorted by confidence (highest to lowest)*")
        
        # Sort bone types by confidence
        sorted_bones = sorted(
            results['bone_confidences'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Display each bone type confidence
        for bone_type, confidence in sorted_bones:
            # Highlight the detected bone type
            if bone_type == results['bone_type']:
                st.markdown(f"**{bone_type.capitalize()} (Detected)**")
            else:
                st.markdown(f"{bone_type.capitalize()}")
            
            color = Visualizer.COLORS.get(bone_type, '#6b7280')
            fig = Visualizer.create_progress_bar(
                "",
                confidence,
                color,
                height=50
            )
            st.plotly_chart(fig, use_container_width=True, key=f"bone_{bone_type}")
    
    @staticmethod
    def display_summary_stats(summary: Dict):
        """
        Display summary statistics for batch analysis
        
        Args:
            summary: Dictionary containing summary statistics
        """
        st.markdown("### 📈 Summary Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total Images", 
                summary.get('total_images', 0)
            )
            st.metric(
                "Successful", 
                summary.get('successful_analyses', 0)
            )
        
        with col2:
            st.metric(
                "Fractures Detected", 
                summary.get('fractures_detected', 0)
            )
            st.metric(
                "No Fractures", 
                summary.get('no_fractures', 0)
            )
        
        with col3:
            st.metric(
                "Fracture Rate", 
                f"{summary.get('fracture_rate', 0)*100:.1f}%"
            )
            st.metric(
                "Avg Confidence", 
                f"{summary.get('avg_bone_confidence', 0)*100:.1f}%"
            )
        
        # Bone type distribution
        if 'bone_type_distribution' in summary:
            st.markdown("#### Bone Type Distribution")
            bone_dist = summary['bone_type_distribution']
            
            fig = go.Figure(data=[
                go.Bar(
                    x=list(bone_dist.keys()),
                    y=list(bone_dist.values()),
                    marker_color=[Visualizer.COLORS.get(b, '#6b7280') for b in bone_dist.keys()]
                )
            ])
            
            fig.update_layout(
                xaxis_title="Bone Type",
                yaxis_title="Count",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def display_error(error_message: str):
        """
        Display error message
        
        Args:
            error_message: Error message to display
        """
        st.error(f"❌ Error: {error_message}")
    
    @staticmethod
    def display_warning(warning_message: str):
        """
        Display warning message
        
        Args:
            warning_message: Warning message to display
        """
        st.warning(f"⚠️ Warning: {warning_message}")
    
    @staticmethod
    def display_info(info_message: str):
        """
        Display info message
        
        Args:
            info_message: Info message to display
        """
        st.info(f"ℹ️ {info_message}")


if __name__ == "__main__":
    # Test visualizer
    print("Visualizer module loaded successfully")