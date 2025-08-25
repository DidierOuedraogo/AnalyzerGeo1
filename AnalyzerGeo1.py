"""
Analyseur Géologique de Veines Minéralisées - Compatible Leapfrog Geo
Auteur: Didier Ouedraogo, P.Geo
Version: 1.2 - Streamlit Cloud Compatible
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import base64
from datetime import datetime
import math
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Analyseur Géologique - Leapfrog Compatible",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #1e40af 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .leapfrog-box {
        background: #f0f9ff;
        border: 2px solid #0ea5e9;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background: #f0fdf4;
        border: 1px solid #22c55e;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fffbeb;
        border: 1px solid #f59e0b;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background: #fef2f2;
        border: 1px solid #ef4444;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialisation des états de session
if 'samples_data' not in st.session_state:
    st.session_state.samples_data = None
if 'structural_data' not in st.session_state:
    st.session_state.structural_data = None
if 'mesh_data' not in st.session_state:
    st.session_state.mesh_data = None
if 'results_data' not in st.session_state:
    st.session_state.results_data = None
if 'leapfrog_intervals' not in st.session_state:
    st.session_state.leapfrog_intervals = None

# Classes et fonctions pour compatibilité Leapfrog
class LeapfrogGeologicalAnalyzer:
    def __init__(self):
        # Colonnes standard Leapfrog pour les échantillons (Assay Table)
        self.leapfrog_assay_columns = {
            'HOLEID': 'Identifiant du forage',
            'FROM': 'Profondeur début (m)',
            'TO': 'Profondeur fin (m)',
            'SAMPLE_ID': 'Identifiant échantillon',
            'Au': 'Teneur or (g/t)',
            'Ag': 'Teneur argent (g/t)',
            'Cu': 'Teneur cuivre (%)',
            'LENGTH': 'Longueur échantillon (m)',
            'RECOVERY': 'Récupération (%)',
            'DENSITY': 'Densité (t/m³)'
        }
        
        # Colonnes standard Leapfrog pour les intervalles (Interval Table)
        self.leapfrog_interval_columns = {
            'HOLEID': 'Identifiant du forage',
            'FROM': 'Profondeur début (m)',
            'TO': 'Profondeur fin (m)',
            'DOMAIN': 'Domaine géologique',
            'ZONE': 'Zone minéralisée',
            'ROCK_TYPE': 'Type de roche',
            'ALTERATION': 'Type d\'altération',
            'MINERALIZATION': 'Type de minéralisation',
            'VEIN_ID': 'Identifiant de veine',
            'CONFIDENCE': 'Niveau de confiance',
            'STRUCTURE_TYPE': 'Type de structure',
            'COMMENTS': 'Commentaires'
        }
        
        # Colonnes pour les données structurales compatibles Leapfrog
        self.leapfrog_structural_columns = {
            'STRUCTURE_ID': 'Identifiant de structure',
            'X': 'Coordonnée X',
            'Y': 'Coordonnée Y',
            'Z': 'Coordonnée Z',
            'STRIKE': 'Direction (°)',
            'DIP': 'Pendage (°)',
            'DIP_DIRECTION': 'Direction de pendage (°)',
            'STRUCTURE_TYPE': 'Type de structure',
            'CONFIDENCE': 'Niveau de confiance',
            'MEASUREMENT_TYPE': 'Type de mesure',
            'VEIN_SET': 'Famille de veines',
            'CAMPAIGN': 'Campagne de mesure',
            'DATE_MEASURED': 'Date de mesure',
            'COMMENTS': 'Commentaires'
        }
    
    def auto_detect_leapfrog_columns(self, df_columns):
        """Auto-détection des colonnes compatibles Leapfrog"""
        mapping = {}
        
        for leapfrog_col, description in self.leapfrog_assay_columns.items():
            # Recherche exacte
            exact_match = next((col for col in df_columns 
                              if col.upper().strip() == leapfrog_col.upper()), None)
            
            if exact_match:
                mapping[leapfrog_col] = exact_match
                continue
            
            # Recherche alternative
            for col in df_columns:
                col_upper = col.upper().strip()
                
                if leapfrog_col == 'HOLEID' and any(word in col_upper for word in ['HOLE', 'DRILL', 'DDH', 'BH', 'SONDAGE', 'ID']):
                    mapping[leapfrog_col] = col
                    break
                elif leapfrog_col == 'FROM' and any(word in col_upper for word in ['FROM', 'DEBUT', 'START', 'DE']):
                    mapping[leapfrog_col] = col
                    break
                elif leapfrog_col == 'TO' and any(word in col_upper for word in ['TO', 'FIN', 'END', 'A']):
                    mapping[leapfrog_col] = col
                    break
                elif leapfrog_col == 'Au' and any(word in col_upper for word in ['AU', 'GOLD', 'OR', 'TENEUR']):
                    mapping[leapfrog_col] = col
                    break
        
        return mapping
    
    def validate_leapfrog_data(self, df, mapping):
        """Validation des données pour Leapfrog"""
        errors = []
        warnings = []
        
        # Vérifier les colonnes obligatoires
        required_cols = ['HOLEID', 'FROM', 'TO']
        for req_col in required_cols:
            if req_col not in mapping or not mapping[req_col]:
                errors.push(f"Colonne obligatoire '{req_col}' manquante")
        
        if errors:
            return None, errors, warnings
        
        # Validation des données
        mapped_df = pd.DataFrame()
        
        for leapfrog_col, source_col in mapping.items():
            if source_col and source_col in df.columns:
                if leapfrog_col in ['FROM', 'TO', 'Au', 'Ag', 'Cu', 'LENGTH', 'RECOVERY', 'DENSITY']:
                    mapped_df[leapfrog_col] = pd.to_numeric(df[source_col], errors='coerce')
                else:
                    mapped_df[leapfrog_col] = df[source_col].astype(str)
        
        # Calculer LENGTH si manquant
        if 'FROM' in mapped_df.columns and 'TO' in mapped_df.columns and 'LENGTH' not in mapped_df.columns:
            mapped_df['LENGTH'] = mapped_df['TO'] - mapped_df['FROM']
            warnings.append("Colonne LENGTH calculée automatiquement")
        
        # Ajouter SAMPLE_ID si manquant
        if 'SAMPLE_ID' not in mapped_df.columns:
            mapped_df['SAMPLE_ID'] = mapped_df['HOLEID'] + '_' + mapped_df['FROM'].astype(str) + '_' + mapped_df['TO'].astype(str)
            warnings.append("SAMPLE_ID généré automatiquement")
        
        # Supprimer les lignes avec des valeurs critiques manquantes
        initial_count = len(mapped_df)
        mapped_df = mapped_df.dropna(subset=['HOLEID', 'FROM', 'TO'])
        final_count = len(mapped_df)
        
        if initial_count > final_count:
            warnings.append(f"{initial_count - final_count} lignes supprimées (valeurs manquantes)")
        
        return mapped_df, errors, warnings
    
    def create_leapfrog_intervals(self, samples_df, params):
        """Création d'intervalles compatibles Leapfrog"""
        if samples_df is None or len(samples_df) == 0:
            return pd.DataFrame()
        
        intervals = []
        min_grade = params.get('min_grade', 0.5)
        max_dilution = params.get('max_dilution', 3.0)
        min_samples = params.get('min_samples', 2)
        
        # Grouper par forage
        for holeid, hole_data in samples_df.groupby('HOLEID'):
            hole_data = hole_data.sort_values('FROM').reset_index(drop=True)
            
            current_interval = None
            zone_id = 1
            
            for idx, sample in hole_data.iterrows():
                grade = sample.get('Au', 0) if pd.notna(sample.get('Au', 0)) else 0
                is_ore = grade >= min_grade
                
                if is_ore:
                    if current_interval is None:
                        current_interval = {'samples': [sample], 'start_idx': idx}
                    else:
                        last_sample = current_interval['samples'][-1]
                        gap = sample['FROM'] - last_sample['TO']
                        
                        if gap <= max_dilution:
                            current_interval['samples'].append(sample)
                        else:
                            if len(current_interval['samples']) >= min_samples:
                                interval = self._create_leapfrog_interval(current_interval, holeid, zone_id)
                                intervals.append(interval)
                                zone_id += 1
                            current_interval = {'samples': [sample], 'start_idx': idx}
                else:
                    if current_interval is not None:
                        if len(current_interval['samples']) >= min_samples:
                            interval = self._create_leapfrog_interval(current_interval, holeid, zone_id)
                            intervals.append(interval)
                            zone_id += 1
                        current_interval = None
            
            # Finaliser le dernier intervalle
            if current_interval is not None and len(current_interval['samples']) >= min_samples:
                interval = self._create_leapfrog_interval(current_interval, holeid, zone_id)
                intervals.append(interval)
        
        if intervals:
            intervals_df = pd.DataFrame(intervals)
            return self._add_leapfrog_metadata(intervals_df, params)
        else:
            return pd.DataFrame()
    
    def _create_leapfrog_interval(self, interval_data, holeid, zone_id):
        """Créer un intervalle au format Leapfrog"""
        samples = pd.DataFrame(interval_data['samples'])
        
        from_depth = samples['FROM'].min()
        to_depth = samples['TO'].max()
        true_width = to_depth - from_depth
        
        total_length = samples['LENGTH'].sum() if 'LENGTH' in samples.columns else true_width
        weighted_grade = (samples['Au'] * samples.get('LENGTH', 1)).sum() / total_length if total_length > 0 else 0
        
        return {
            'HOLEID': holeid,
            'FROM': round(from_depth, 2),
            'TO': round(to_depth, 2),
            'DOMAIN': self._classify_domain(weighted_grade),
            'ZONE': f"ZONE_{zone_id:03d}",
            'VEIN_ID': f"VEIN_{zone_id:03d}",
            'CONFIDENCE': round(min(0.95, 0.5 + len(samples) * 0.1), 2),
            'STRUCTURE_TYPE': 'VEIN',
            'TRUE_WIDTH': round(true_width, 2),
            'WEIGHTED_GRADE': round(weighted_grade, 3),
            'SAMPLE_COUNT': len(samples),
            'METAL_CONTENT': round(weighted_grade * true_width, 3)
        }
    
    def _classify_domain(self, grade):
        """Classification en domaine géologique"""
        if grade >= 5.0:
            return 'HIGH_GRADE'
        elif grade >= 2.0:
            return 'ORE_ZONE'
        elif grade >= 0.5:
            return 'LOW_GRADE'
        else:
            return 'WASTE'
    
    def _add_leapfrog_metadata(self, intervals_df, params):
        """Ajouter les métadonnées Leapfrog"""
        if len(intervals_df) == 0:
            return intervals_df
        
        intervals_df['CAMPAIGN'] = params.get('campaign', 'DEFAULT_CAMPAIGN')
        intervals_df['DATE_CREATED'] = datetime.now().strftime('%Y-%m-%d')
        intervals_df['CREATED_BY'] = 'GEOLOGICAL_ANALYZER'
        
        return intervals_df
    
    def export_leapfrog_format(self, data_type, df, filename_prefix):
        """Export au format standard Leapfrog"""
        if df is None or len(df) == 0:
            return None
        
        header_lines = [
            f"# Leapfrog Geo Compatible Export - {data_type}",
            f"# Generated by: Geological Analyzer v1.2",
            f"# Author: Didier Ouedraogo, P.Geo",
            f"# Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"# Total Records: {len(df)}",
            f"# Coordinate System: UTM Zone 18N (EPSG:32618)",
            f"# Units: Meters (length), PPM (grade)",
            "#"
        ]
        
        csv_buffer = io.StringIO()
        
        for line in header_lines:
            csv_buffer.write(line + '\n')
        
        df.to_csv(csv_buffer, index=False, float_format='%.3f')
        
        csv_content = csv_buffer.getvalue()
        csv_buffer.close()
        
        return csv_content

def generate_leapfrog_demo_data():
    """Génération de données de démonstration compatibles Leapfrog"""
    np.random.seed(42)
    
    # Paramètres du gisement aurifère
    deposit_center = {'x': 450000, 'y': 5100000, 'z': 350}
    
    # Systèmes de veines aurifères
    vein_systems = [
        {
            'id': 'MAIN_LODE', 'strike': 45, 'dip': 65, 'center_x': 450000, 'center_y': 5100000,
            'avg_grade': 8.5, 'width': 2.5, 'continuity': 0.85
        },
        {
            'id': 'HANGING_WALL', 'strike': 50, 'dip': 70, 'center_x': 450150, 'center_y': 5100100,
            'avg_grade': 4.2, 'width': 1.5, 'continuity': 0.65
        }
    ]
    
    # Génération des échantillons
    samples = []
    
    for hole_num in range(1, 51):  # 50 forages
        hole_id = f"DDH{hole_num:04d}"
        
        grid_x = ((hole_num - 1) % 10) * 40
        grid_y = ((hole_num - 1) // 10) * 40
        hole_x = deposit_center['x'] - 200 + grid_x + np.random.normal(0, 5)
        hole_y = deposit_center['y'] - 200 + grid_y + np.random.normal(0, 5)
        hole_z = deposit_center['z'] + np.random.uniform(-15, 15)
        
        max_depth = np.random.uniform(200, 400)
        sample_length = 1.5
        num_samples = int(max_depth / sample_length)
        
        for i in range(num_samples):
            from_depth = i * sample_length
            to_depth = (i + 1) * sample_length
            
            sample_x = hole_x + np.random.normal(0, 1)
            sample_y = hole_y + np.random.normal(0, 1)
            sample_z = hole_z - (from_depth + to_depth) / 2
            
            au_grade = np.random.lognormal(np.log(0.05), 0.5)
            ag_grade = au_grade * np.random.uniform(5, 20)
            cu_grade = np.random.uniform(0.01, 0.1)
            
            # Influence des veines
            for vein in vein_systems:
                distance_to_vein = np.sqrt(
                    (sample_x - vein['center_x'])**2 + 
                    (sample_y - vein['center_y'])**2
                )
                
                spatial_influence = np.exp(-distance_to_vein / 80)
                intersection_prob = spatial_influence * vein['continuity']
                
                if np.random.random() < intersection_prob * 0.4:
                    grade_factor = vein['avg_grade'] * np.random.uniform(0.7, 1.3)
                    au_grade = max(au_grade, grade_factor)
                    ag_grade = au_grade * np.random.uniform(10, 30)
            
            au_grade = min(au_grade, 100.0)
            ag_grade = min(ag_grade, 2000.0)
            
            samples.append({
                'HOLEID': hole_id,
                'FROM': round(from_depth, 1),
                'TO': round(to_depth, 1),
                'SAMPLE_ID': f"{hole_id}_{from_depth:06.1f}_{to_depth:06.1f}",
                'Au': round(au_grade, 3),
                'Ag': round(ag_grade, 2),
                'Cu': round(cu_grade, 3),
                'LENGTH': sample_length,
                'RECOVERY': round(np.random.uniform(85, 98), 1),
                'DENSITY': round(np.random.uniform(2.5, 2.9), 2),
                'XCOLLAR': round(hole_x, 2),
                'YCOLLAR': round(hole_y, 2),
                'ZCOLLAR': round(hole_z, 2)
            })
    
    # Données structurales
    structural_data = []
    for i, vein in enumerate(vein_systems):
        for j in range(5):
            measurement_x = vein['center_x'] + np.random.uniform(-100, 100)
            measurement_y = vein['center_y'] + np.random.uniform(-100, 100)
            measurement_z = deposit_center['z'] - np.random.uniform(50, 300)
            
            strike_var = vein['strike'] + np.random.normal(0, 5)
            dip_var = vein['dip'] + np.random.normal(0, 3)
            
            structural_data.append({
                'STRUCTURE_ID': f"{vein['id']}_M{j+1:02d}",
                'X': round(measurement_x, 2),
                'Y': round(measurement_y, 2),
                'Z': round(measurement_z, 2),
                'STRIKE': round(strike_var, 1),
                'DIP': round(dip_var, 1),
                'DIP_DIRECTION': round((strike_var + 90) % 360, 1),
                'STRUCTURE_TYPE': 'VEIN',
                'CONFIDENCE': round(np.random.uniform(0.7, 0.95), 2),
                'MEASUREMENT_TYPE': 'ORIENTED_CORE',
                'VEIN_SET': vein['id']
            })
    
    # Mesh géologique
    mesh_data = []
    fault_strike = 120
    fault_dip = 80
    
    for i in range(100):
        along_strike = np.random.uniform(-300, 300)
        along_dip = np.random.uniform(-200, 200)
        
        x = deposit_center['x'] + along_strike * np.cos(np.radians(fault_strike))
        y = deposit_center['y'] + along_strike * np.sin(np.radians(fault_strike))
        z = deposit_center['z'] - along_dip * np.sin(np.radians(fault_dip))
        
        mesh_data.append({
            'x': round(x, 2),
            'y': round(y, 2),
            'z': round(z, 2),
            'structure_id': 'MAJOR_FAULT'
        })
    
    return pd.DataFrame(samples), pd.DataFrame(structural_data), pd.DataFrame(mesh_data)

def create_leapfrog_qaqc_plots(df):
    """Création de graphiques QA/QC compatibles Leapfrog - CORRIGÉ"""
    
    # Distribution des teneurs (log-scale pour l'or)
    fig1 = px.histogram(
        df, 
        x='Au', 
        nbins=50,
        title="Distribution des Teneurs en Or (Au)",
        labels={'Au': 'Teneur Au (ppm)', 'count': 'Fréquence'},
        log_x=True
    )
    fig1.add_vline(x=0.5, line_dash="dash", line_color="red", 
                   annotation_text="Cut-off 0.5 ppm")
    
    # Corrélation Au-Ag SANS trendline OLS
    sample_df = df.sample(min(1000, len(df))) if len(df) > 1000 else df
    
    fig2 = px.scatter(
        sample_df, 
        x='Au', y='Ag',
        title="Corrélation Au-Ag",
        labels={'Au': 'Teneur Au (ppm)', 'Ag': 'Teneur Ag (ppm)'}
    )
    
    # Ajouter une ligne de tendance manuelle avec sklearn
    if len(sample_df) > 1 and sample_df['Au'].std() > 0:
        try:
            # Préparer les données pour la régression
            X = sample_df['Au'].values.reshape(-1, 1)
            y = sample_df['Ag'].values
            
            # Supprimer les valeurs NaN
            mask = ~(np.isnan(X.flatten()) | np.isnan(y))
            X_clean = X[mask]
            y_clean = y[mask]
            
            if len(X_clean) > 1:
                # Régression linéaire
                reg = LinearRegression()
                reg.fit(X_clean, y_clean)
                
                # Ligne de tendance
                x_range = np.linspace(X_clean.min(), X_clean.max(), 100)
                y_pred = reg.predict(x_range.reshape(-1, 1))
                
                fig2.add_trace(go.Scatter(
                    x=x_range.flatten(),
                    y=y_pred,
                    mode='lines',
                    name=f'Tendance (R²={reg.score(X_clean, y_clean):.3f})',
                    line=dict(color='red', dash='dash')
                ))
        except Exception:
            pass  # Si erreur, on continue sans la ligne de tendance
    
    # Distribution spatiale des teneurs
    fig3 = None
    if 'XCOLLAR' in df.columns and 'YCOLLAR' in df.columns:
        hole_summary = df.groupby('HOLEID').agg({
            'XCOLLAR': 'first',
            'YCOLLAR': 'first',
            'Au': 'mean'
        }).reset_index()
        
        fig3 = px.scatter(
            hole_summary,
            x='XCOLLAR', y='YCOLLAR',
            color='Au',
            size='Au',
            title="Distribution Spatiale des Teneurs",
            labels={'XCOLLAR': 'X (UTM)', 'YCOLLAR': 'Y (UTM)', 'Au': 'Au (ppm)'},
            color_continuous_scale='Viridis'
        )
    
    # Profil de teneur par profondeur
    try:
        depth_bins = pd.cut(df['FROM'], bins=20)
        depth_profile = df.groupby(depth_bins)['Au'].mean().reset_index()
        depth_profile['Depth_Mid'] = depth_profile['FROM'].apply(lambda x: x.mid if hasattr(x, 'mid') else 0)
        
        fig4 = px.line(
            depth_profile,
            x='Depth_Mid', y='Au',
            title="Profil de Teneur par Profondeur",
            labels={'Depth_Mid': 'Profondeur (m)', 'Au': 'Teneur Moyenne Au (ppm)'}
        )
    except Exception:
        # Fallback simple
        fig4 = px.scatter(
            df.sample(min(500, len(df))),
            x='FROM', y='Au',
            title="Teneur vs Profondeur",
            labels={'FROM': 'Profondeur (m)', 'Au': 'Teneur Au (ppm)'}
        )
    
    return fig1, fig2, fig3, fig4

def create_compass_plot(strike, dip, vein_id):
    """Création d'un diagramme en boussole"""
    fig = go.Figure()
    
    # Cercle extérieur
    theta = np.linspace(0, 2*np.pi, 100)
    x_circle = np.cos(theta)
    y_circle = np.sin(theta)
    
    fig.add_trace(go.Scatter(
        x=x_circle, y=y_circle,
        mode='lines',
        line=dict(color='lightgray', width=2),
        showlegend=False
    ))
    
    # Directions cardinales
    fig.add_annotation(x=0, y=1.1, text="N", showarrow=False, font=dict(size=14, color='red'))
    fig.add_annotation(x=1.1, y=0, text="E", showarrow=False, font=dict(size=14))
    fig.add_annotation(x=0, y=-1.1, text="S", showarrow=False, font=dict(size=14))
    fig.add_annotation(x=-1.1, y=0, text="W", showarrow=False, font=dict(size=14))
    
    # Ligne de strike
    strike_rad = np.radians(strike)
    x_strike = [np.sin(strike_rad), -np.sin(strike_rad)]
    y_strike = [np.cos(strike_rad), -np.cos(strike_rad)]
    
    fig.add_trace(go.Scatter(
        x=x_strike, y=y_strike,
        mode='lines',
        line=dict(color='blue', width=4),
        name=f'Strike: {strike}°'
    ))
    
    # Indicateur de pendage
    dip_length = 0.7
    dip_x = dip_length * np.sin(strike_rad + np.pi/2)
    dip_y = dip_length * np.cos(strike_rad + np.pi/2)
    
    fig.add_trace(go.Scatter(
        x=[0, dip_x], y=[0, dip_y],
        mode='lines+markers',
        line=dict(color='red', width=3, dash='dash'),
        marker=dict(size=8, color='red'),
        name=f'Dip: {dip}°'
    ))
    
    fig.update_layout(
        title=f"Mesure Structurale: {vein_id}",
        xaxis=dict(range=[-1.3, 1.3], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-1.3, 1.3], showgrid=False, zeroline=False, showticklabels=False),
        aspectratio=dict(x=1, y=1),
        height=400,
        showlegend=True
    )
    
    return fig

def create_stereonet_plot(structural_df):
    """Création d'un stéréonet"""
    fig = go.Figure()
    
    # Cercle de référence
    theta = np.linspace(0, 2*np.pi, 100)
    fig.add_trace(go.Scatter(
        x=np.cos(theta), y=np.sin(theta),
        mode='lines',
        line=dict(color='lightgray', width=2),
        showlegend=False
    ))
    
    # Grille de référence
    for r in [0.25, 0.5, 0.75]:
        fig.add_trace(go.Scatter(
            x=r*np.cos(theta), y=r*np.sin(theta),
            mode='lines',
            line=dict(color='lightgray', width=1),
            showlegend=False,
            opacity=0.5
        ))
    
    # Points de mesures
    for _, measure in structural_df.iterrows():
        strike_rad = np.radians(measure['STRIKE'])
        dip_rad = np.radians(measure['DIP'])
        
        # Projection stéréographique
        r = np.tan(dip_rad / 2)
        if r <= 1:  # Vérifier que le point est dans le cercle
            x = r * np.sin(strike_rad)
            y = r * np.cos(strike_rad)
            
            confidence = measure.get('CONFIDENCE', 0.8)
            color = 'green' if confidence > 0.8 else 'orange' if confidence > 0.6 else 'red'
            
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers+text',
                marker=dict(size=10, color=color),
                text=measure['STRUCTURE_ID'] if 'STRUCTURE_ID' in measure else 'STRUCT',
                textposition='top center',
                name=f"{measure.get('STRUCTURE_ID', 'STRUCT')} ({measure['STRIKE']:.0f}°/{measure['DIP']:.0f}°)"
            ))
    
    fig.update_layout(
        title="Projection Stéréographique",
        xaxis=dict(range=[-1.2, 1.2], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-1.2, 1.2], showgrid=False, zeroline=False, showticklabels=False),
        aspectratio=dict(x=1, y=1),
        height=500
    )
    
    return fig

def create_rose_diagram(structural_df):
    """Création d'un diagramme en rosace"""
    # Grouper les directions par secteurs de 15°
    bin_size = 15
    bins = np.arange(0, 361, bin_size)
    strikes = structural_df['STRIKE'].values
    
    hist, bin_edges = np.histogram(strikes, bins=bins)
    
    # Créer le diagramme polaire
    fig = go.Figure()
    
    for i, count in enumerate(hist):
        if count > 0:
            theta_start = bin_edges[i]
            theta_end = bin_edges[i + 1]
            theta_mid = (theta_start + theta_end) / 2
            
            fig.add_trace(go.Barpolar(
                r=[count],
                theta=[theta_mid],
                width=[bin_size],
                name=f"{theta_start}°-{theta_end}°",
                showlegend=False
            ))
    
    fig.update_layout(
        title="Diagramme en Rosace - Directions",
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max(hist) if len(hist) > 0 else 1]),
            angularaxis=dict(direction='clockwise', rotation=90)
        ),
        height=500
    )
    
    return fig

def main():
    # En-tête principal avec badge Leapfrog
    st.markdown("""
    <div class="main-header">
        <h1>⛏️ Analyseur Géologique - Compatible Leapfrog Geo</h1>
        <h2>🎯 Export Direct vers Leapfrog | Standards Industriels</h2>
        <p style="margin-top: 1rem; opacity: 0.9;">
            Version Streamlit Cloud - Optimisée et Compatible
        </p>
        <div style="background: rgba(255,255,255,0.2); border-radius: 0.5rem; padding: 1rem; margin-top: 1.5rem;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h4>👨‍🔬 Auteur</h4>
                    <h3>Didier Ouedraogo, P.Geo</h3>
                    <p>Géologue Professionnel | Spécialiste Leapfrog Geo & Modélisation 3D</p>
                </div>
                <div style="text-align: right;">
                    <p>📅 Version: 1.2 - Streamlit Cloud</p>
                    <p>🎯 Standards: Leapfrog Geo 2024</p>
                    <p>Date: {datetime.now().strftime('%B %Y')}</p>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Badge de compatibilité Leapfrog
    st.markdown("""
    <div class="leapfrog-box">
        <h4>🎯 Compatibilité Leapfrog Geo Certifiée</h4>
        <p><strong>✅ Formats supportés:</strong> Assay Table, Interval Table, Structural Data</p>
        <p><strong>✅ Standards respectés:</strong> Nomenclature Leapfrog, unités métriques, CRS standardisé</p>
        <p><strong>✅ Import direct:</strong> Fichiers prêts pour import dans Leapfrog Geo</p>
        <p><strong>✅ Streamlit Cloud:</strong> Optimisé pour déploiement cloud</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialiser l'analyseur Leapfrog
    analyzer = LeapfrogGeologicalAnalyzer()
    
    # Sidebar pour la navigation
    st.sidebar.title("🧭 Navigation Leapfrog")
    tab_selected = st.sidebar.selectbox(
        "Sélectionner une section:",
        ["📤 Import & Démonstration", "🔗 Mapping Leapfrog", "⚙️ Analyse & Intervalles", "📊 Export Leapfrog"]
    )
    
    # Section Import & Démonstration
    if tab_selected == "📤 Import & Démonstration":
        st.header("📤 Import Multi-Format avec Données Demo Leapfrog")
        
        # Génération de données demo compatibles Leapfrog
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div class="success-box">
                <h4>🚀 Données de Démonstration Leapfrog</h4>
                <p>Génère un gisement aurifère réaliste avec:</p>
                <ul>
                    <li>50 forages DDH format standard</li>
                    <li>~3000 échantillons Au-Ag-Cu</li>
                    <li>2 systèmes de veines aurifères</li>
                    <li>Données structurales complètes</li>
                    <li>Mesh de faille majeure</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if st.button("⚡ Générer Dataset Leapfrog", type="primary"):
                with st.spinner("Génération du gisement aurifère..."):
                    samples_demo, structural_demo, mesh_demo = generate_leapfrog_demo_data()
                    st.session_state.samples_data = samples_demo
                    st.session_state.structural_data = structural_demo
                    st.session_state.mesh_data = mesh_demo
                    st.success(f"""
                    ✅ **Dataset Leapfrog Généré!**
                    
                    📊 **Données créées:**
                    - {len(samples_demo):,} échantillons Au-Ag-Cu
                    - {samples_demo['HOLEID'].nunique()} forages DDH
                    - {len(structural_demo)} mesures structurales
                    - {len(mesh_demo)} points de mesh de faille
                    """)
        
        # Statistiques et QA/QC
        if st.session_state.samples_data is not None:
            st.markdown("---")
            st.subheader("📊 QA/QC et Statistiques Leapfrog")
            
            samples_df = st.session_state.samples_data
            
            # Métriques principales
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("📊 Échantillons", f"{len(samples_df):,}")
            with col2:
                st.metric("🏗️ Forages", samples_df['HOLEID'].nunique())
            with col3:
                avg_au = samples_df['Au'].mean()
                st.metric("🥇 Au Moyen", f"{avg_au:.2f} ppm")
            with col4:
                max_au = samples_df['Au'].max()
                st.metric("⭐ Au Max", f"{max_au:.1f} ppm")
            with col5:
                mineralized = len(samples_df[samples_df['Au'] >= 0.5])
                pct_min = mineralized / len(samples_df) * 100
                st.metric("💎 Minéralisé", f"{pct_min:.1f}%")
            
            # Graphiques QA/QC
            st.subheader("📈 Graphiques QA/QC Leapfrog")
            
            with st.spinner("Génération des graphiques QA/QC..."):
                fig1, fig2, fig3, fig4 = create_leapfrog_qaqc_plots(samples_df)
                
                tab1, tab2, tab3, tab4 = st.tabs(["📊 Distribution Au", "🔗 Corrélation Au-Ag", "🗺️ Spatial", "📉 Profondeur"])
                
                with tab1:
                    st.plotly_chart(fig1, use_container_width=True)
                
                with tab2:
                    st.plotly_chart(fig2, use_container_width=True)
                    correlation = samples_df['Au'].corr(samples_df['Ag'])
                    st.metric("Corrélation Au-Ag", f"{correlation:.3f}")
                
                with tab3:
                    if fig3:
                        st.plotly_chart(fig3, use_container_width=True)
                    else:
                        st.warning("⚠️ Coordonnées spatiales non disponibles")
                
                with tab4:
                    st.plotly_chart(fig4, use_container_width=True)
    
    # Section Mapping Leapfrog
    elif tab_selected == "🔗 Mapping Leapfrog":
        st.header("🔗 Mapping des Colonnes Format Leapfrog")
        
        st.markdown("""
        <div class="leapfrog-box">
            <h4>🎯 Auto-détection Format Leapfrog</h4>
            <p>Le système détecte automatiquement les colonnes selon les standards Leapfrog:</p>
            <ul>
                <li><strong>HOLEID:</strong> Identifiant unique du forage</li>
                <li><strong>FROM/TO:</strong> Intervalles en mètres</li>
                <li><strong>Au/Ag/Cu:</strong> Teneurs en ppm ou %</li>
                <li><strong>SAMPLE_ID:</strong> Identifiant échantillon</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.samples_data is not None:
            st.success("✅ Données déjà disponibles au format Leapfrog!")
            st.dataframe(st.session_state.samples_data.head(10))
        else:
            st.warning("⚠️ Générez d'abord des données de démonstration")
    
    # Section Analyse & Intervalles
    elif tab_selected == "⚙️ Analyse & Intervalles":
        st.header("⚙️ Création d'Intervalles Leapfrog")
        
        if st.session_state.samples_data is None:
            st.warning("⚠️ Générez d'abord des données de démonstration")
            return
        
        samples_df = st.session_state.samples_data
        
        st.markdown("""
        <div class="leapfrog-box">
            <h4>🎯 Création d'Intervalles pour Leapfrog Geo</h4>
            <p>Génération d'intervalles minéralisés selon les standards Leapfrog avec:</p>
            <ul>
                <li><strong>DOMAIN:</strong> Classification automatique des domaines</li>
                <li><strong>CONFIDENCE:</strong> Score de confiance calculé</li>
                <li><strong>ZONE:</strong> Identification des zones minéralisées</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Paramètres d'analyse
        st.subheader("🔧 Paramètres d'Analyse")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_grade = st.number_input("Cut-off Au (ppm)", min_value=0.0, value=0.5, step=0.1)
        with col2:
            max_dilution = st.number_input("Dilution max (m)", min_value=0.1, value=3.0, step=0.5)
        with col3:
            min_samples = st.number_input("Échantillons min", min_value=1, value=2)
        
        # Aperçu des critères
        qualifying_samples = samples_df[samples_df['Au'] >= min_grade]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Échantillons Qualifiés", f"{len(qualifying_samples):,}")
        with col2:
            st.metric("Pourcentage", f"{len(qualifying_samples)/len(samples_df)*100:.1f}%")
        with col3:
            affected_holes = qualifying_samples['HOLEID'].nunique()
            st.metric("Forages Affectés", affected_holes)
        
        # Analyse
        analysis_params = {
            'min_grade': min_grade,
            'max_dilution': max_dilution,
            'min_samples': min_samples,
            'campaign': 'DEMO_ANALYSIS_2024'
        }
        
        if st.button("🚀 Créer Intervalles Leapfrog", type="primary"):
            with st.spinner("Création des intervalles..."):
                intervals_df = analyzer.create_leapfrog_intervals(samples_df, analysis_params)
                
                if len(intervals_df) > 0:
                    st.session_state.leapfrog_intervals = intervals_df
                    
                    st.success(f"""
                    ✅ **Intervalles Créés!**
                    - {len(intervals_df)} intervalles générés
                    - {intervals_df['HOLEID'].nunique()} forages avec intervalles
                    - {len(intervals_df['DOMAIN'].unique())} domaines géologiques
                    """)
                    
                    # Aperçu des résultats
                    st.subheader("📊 Aperçu des Intervalles")
                    st.dataframe(intervals_df.head(20))
                    
                    # Statistiques par domaine
                    domain_stats = intervals_df.groupby('DOMAIN').agg({
                        'HOLEID': 'count',
                        'TRUE_WIDTH': 'sum',
                        'WEIGHTED_GRADE': 'mean'
                    }).round(2)
                    
                    st.subheader("📋 Statistiques par Domaine")
                    st.dataframe(domain_stats)
                else:
                    st.error("❌ Aucun intervalle généré. Ajustez les paramètres.")
    
    # Section Export Leapfrog
    elif tab_selected == "📊 Export Leapfrog":
        st.header("📊 Export Format Leapfrog Geo")
        
        st.markdown("""
        <div class="leapfrog-box">
            <h4>🎯 Export Direct vers Leapfrog Geo</h4>
            <p>Formats d'export compatibles Leapfrog Geo 2024:</p>
            <ul>
                <li><strong>Assay Table:</strong> Données d'échantillons</li>
                <li><strong>Interval Table:</strong> Intervalles minéralisés</li>
                <li><strong>Structural Data:</strong> Mesures structurales</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Export Assay Table
        if st.session_state.samples_data is not None:
            st.subheader("📊 Export Assay Table")
            
            samples_df = st.session_state.samples_data
            st.info(f"Prêt à exporter {len(samples_df):,} échantillons de {samples_df['HOLEID'].nunique()} forages")
            
            if st.button("📥 Exporter Assay Table"):
                csv_content = analyzer.export_leapfrog_format("Assay_Table", samples_df, "assay")
                
                st.download_button(
                    label="💾 Télécharger Assay Table",
                    data=csv_content,
                    file_name=f"leapfrog_assay_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        # Export Interval Table
        if st.session_state.leapfrog_intervals is not None:
            st.subheader("📋 Export Interval Table")
            
            intervals_df = st.session_state.leapfrog_intervals
            st.info(f"Prêt à exporter {len(intervals_df)} intervalles de {len(intervals_df['DOMAIN'].unique())} domaines")
            
            if st.button("📥 Exporter Interval Table"):
                csv_content = analyzer.export_leapfrog_format("Interval_Table", intervals_df, "intervals")
                
                st.download_button(
                    label="💾 Télécharger Interval Table",
                    data=csv_content,
                    file_name=f"leapfrog_intervals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        # Export Structural Data
        if st.session_state.structural_data is not None:
            st.subheader("📐 Export Structural Data")
            
            structural_df = st.session_state.structural_data
            st.info(f"Prêt à exporter {len(structural_df)} mesures structurales")
            
            if st.button("📥 Exporter Structural Data"):
                csv_content = analyzer.export_leapfrog_format("Structural_Data", structural_df, "structural")
                
                st.download_button(
                    label="💾 Télécharger Structural Data",
                    data=csv_content,
                    file_name=f"leapfrog_structural_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        # Instructions d'import
        st.markdown("---")
        st.markdown("""
        ### 📋 Instructions Import Leapfrog
        
        **🎯 Ordre d'import recommandé:**
        1. **Assay Table** → Data Files → Samples/Assays
        2. **Interval Table** → Data Files → Intervals
        3. **Structural Data** → Data Files → Structural Data
        
        **⚙️ Paramètres d'import:**
        - Système coordonnées: UTM Zone 18N
        - Unités: Mètres, PPM
        - Séparateur: Virgule (,)
        """)

if __name__ == "__main__":
    main()