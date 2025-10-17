import os
import numpy as np
import pandas as pd
import joblib
import json
from typing import Dict, List, Tuple, Optional, Union
import wfdb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import time


class PhysioNetBPPredictor:
    """
    PhysioNet 데이터를 사용한 혈압 예측 시스템
    
    주요 기능:
    - 데이터 로드 및 전처리
    - 특성 추출 및 준비
    - 모델 학습 및 예측
    - 특성 중요도 분석
    - 모델 저장 및 로드
    """
    
    def __init__(self, data_dir: str = 'data'):
        """
        초기화
        
        Args:
            data_dir: PhysioNet 데이터 디렉토리 경로
        """
        self.data_dir = data_dir
        self.models = {}
        self.scaler = None
        self.feature_names = []
        self.metadata = {}
        
        print("🩺 PhysioNetBPPredictor 초기화 완료")
    
    # ================================================================
    # 데이터 로드 관련 메서드
    # ================================================================
    
    def extract_patient_info(self, record, record_name: str) -> Optional[Dict]:
        """
        WFDB 레코드에서 환자 정보 추출
        
        Args:
            record: WFDB record 객체
            record_name: 레코드 이름
        
        Returns:
            환자 정보 딕셔너리 또는 None (실패 시)
        """
        try:
            patient_data = {
                'patient_id': record_name,
                'sampling_rate': record.fs,
                'signal_length': record.sig_len
            }
            
            if record.p_signal is not None and len(record.p_signal) > 0:
                for i, sig_name in enumerate(record.sig_name):
                    signal_data = record.p_signal[:, i]
                    patient_data[f'{sig_name}_mean'] = np.mean(signal_data)
                    patient_data[f'{sig_name}_std'] = np.std(signal_data)
                    patient_data[f'{sig_name}_max'] = np.max(signal_data)
                    patient_data[f'{sig_name}_min'] = np.min(signal_data)
            
            if hasattr(record, 'comments') and record.comments:
                for comment in record.comments:
                    if ':' in comment:
                        key, value = comment.split(':', 1)
                        patient_data[key.strip()] = value.strip()
            
            return patient_data
            
        except Exception as e:
            print(f"⚠️ 환자 정보 추출 실패 ({record_name}): {str(e)}")
            return None
    
    def load_all_patient_data(self, max_records: Optional[int] = None, 
                            extract_features: bool = True) -> pd.DataFrame:
        """
        모든 환자 데이터를 로드
        
        Args:
            max_records: 읽을 최대 레코드 수 (None이면 전체)
            extract_features: True면 신호 특성 추출, False면 메타데이터만
        
        Returns:
            모든 환자 데이터가 포함된 DataFrame
        """
        print(f"🚀 모든 환자 데이터 로드 시작: {self.data_dir}")
        
        hea_files = sorted([f for f in os.listdir(self.data_dir) if f.endswith('.hea')])
        
        if not hea_files:
            raise FileNotFoundError(f"❌ {self.data_dir}에 .hea 파일이 없습니다.")
        
        if max_records:
            hea_files = hea_files[:max_records]
        
        print(f"📊 처리할 환자 데이터: {len(hea_files)}개")
        
        all_patient_data = []
        success_count = 0
        fail_count = 0
        
        for idx, hea_file in enumerate(hea_files, 1):
            try:
                record_name = hea_file.replace('.hea', '')
                record_path = os.path.join(self.data_dir, record_name)
                record = wfdb.rdrecord(record_path)
                
                if extract_features:
                    patient_data = self.extract_patient_info(record, record_name)
                    if patient_data:
                        all_patient_data.append(patient_data)
                        success_count += 1
                else:
                    patient_data = {
                        'record_name': record_name,
                        'sampling_rate': record.fs,
                        'signal_length': record.sig_len,
                        'n_signals': record.n_sig,
                        'signal_names': ', '.join(record.sig_name),
                        'units': ', '.join(record.units)
                    }
                    all_patient_data.append(patient_data)
                    success_count += 1
                
                if idx % 100 == 0:
                    print(f"  ⏳ 진행 중... {idx}/{len(hea_files)} ({idx/len(hea_files)*100:.1f}%)")
                    
            except Exception as e:
                fail_count += 1
                if fail_count <= 5:
                    print(f"  ⚠️ {hea_file} 로드 실패: {str(e)}")
                continue
        
        df = pd.DataFrame(all_patient_data)
        
        print(f"\n✅ 데이터 로드 완료!")
        print(f"   - 성공: {success_count}개")
        print(f"   - 실패: {fail_count}개")
        print(f"   - 컬럼 수: {len(df.columns)}개")
        print(f"   - 데이터 shape: {df.shape}")
        
        return df
    
    # ================================================================
    # 전처리 관련 메서드
    # ================================================================
    
    def preprocess_data(self, df: pd.DataFrame, target_column: str = 'NIBP_mean',
                       remove_target_outliers: bool = True) -> pd.DataFrame:
        """
        PhysioNet 데이터 전처리
        
        Args:
            df: 원본 DataFrame
            target_column: 예측 대상 컬럼
            remove_target_outliers: 타겟 변수의 이상치 제거 여부
        
        Returns:
            전처리된 DataFrame
        """
        print("=" * 60)
        print("🔄 PhysioNet 데이터 전처리 시작")
        print("=" * 60)
        
        df = df.copy()
        print(f"\n📊 초기 데이터 shape: {df.shape}")
        
        # 결측값 처리
        print("\n2️⃣ 결측값 처리")
        missing_before = df.isnull().sum().sum()
        print(f"   - 처리 전 결측값: {missing_before}개")
        
        if missing_before > 0:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if col != 'patient_id':
                    mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                    df[col] = df[col].fillna(mode_val)
            
            print(f"   - 처리 후 결측값: {df.isnull().sum().sum()}개")
        
        # 무한대 값 및 이상치 처리
        print("\n3️⃣ 무한대 값 및 이상치 처리")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in ['patient_id', 'sampling_rate', 'signal_length']:
                continue
            
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                df[col] = df[col].fillna(df[col].median())
            
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR > 0:
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        # 타겟 변수 이상치 제거
        if remove_target_outliers and target_column in df.columns:
            print(f"\n4️⃣ 타겟 변수 ({target_column}) 이상치 제거")
            before_count = len(df)
            
            if 'NIBP' in target_column:
                valid_range = (40, 200)
                df = df[(df[target_column] >= valid_range[0]) & 
                       (df[target_column] <= valid_range[1])]
                removed = before_count - len(df)
                if removed > 0:
                    print(f"   - 비정상 범위 제거: {removed}개")
        
        # 특성 엔지니어링
        print("\n5️⃣ 특성 엔지니어링")
        
        if 'ECG1_mean' in df.columns and 'ECG2_mean' in df.columns:
            df['ECG_diff_mean'] = abs(df['ECG1_mean'] - df['ECG2_mean'])
            df['ECG_avg_mean'] = (df['ECG1_mean'] + df['ECG2_mean']) / 2
            print("   - ECG 신호 차이 및 평균 계산 ✅")
        
        if 'NIBP_std' in df.columns and 'NIBP_mean' in df.columns:
            df['NIBP_cv'] = df['NIBP_std'] / (df['NIBP_mean'] + 1e-8)
            print("   - NIBP 변동계수 계산 ✅")
        
        if 'signal_length' in df.columns and 'sampling_rate' in df.columns:
            df['duration_minutes'] = df['signal_length'] / df['sampling_rate'] / 60
            print("   - 신호 길이(분) 계산 ✅")
        
        print("\n" + "=" * 60)
        print("✅ 전처리 완료")
        print("=" * 60)
        print(f"📊 최종 데이터 shape: {df.shape}")
        
        return df
    
    # ================================================================
    # 특성 준비 관련 메서드
    # ================================================================
    
    def prepare_features(self, df: pd.DataFrame, target_col: str = 'NIBP_mean',
                        exclude_nibp_features: bool = True) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        모델 훈련에 사용할 특성(X)과 타겟 변수(y) 분리
        
        Args:
            df: 전처리된 DataFrame
            target_col: 타겟 변수 컬럼명
            exclude_nibp_features: NIBP 관련 특성 제외 여부
        
        Returns:
            X (DataFrame): 독립 변수
            y (Series): 종속 변수
            feature_names (list): 특성 이름 리스트
        """
        print("=" * 60)
        print("🎯 특성 준비 시작")
        print("=" * 60)
        
        df = df.copy()
        
        # 타겟 변수 확인
        if target_col not in df.columns:
            nibp_candidates = [col for col in df.columns if 'NIBP' in col]
            if 'NIBP_max' in nibp_candidates:
                target_col = 'NIBP_max'
            elif 'NIBP_mean' in nibp_candidates:
                target_col = 'NIBP_mean'
            else:
                raise ValueError(f"❌ 타겟 변수를 찾을 수 없습니다.")
        
        y = df[target_col]
        print(f"\n타겟 변수: {target_col}")
        print(f"   - 평균: {y.mean():.2f}")
        print(f"   - 표준편차: {y.std():.2f}")
        print(f"   - 범위: {y.min():.2f} ~ {y.max():.2f}")
        
        # 제외할 컬럼 선택
        exclude_cols = []
        
        id_cols = [col for col in df.columns if any(x in col.lower() 
                  for x in ['id', 'patient', 'record_name'])]
        exclude_cols.extend(id_cols)
        exclude_cols.append(target_col)
        
        if exclude_nibp_features and 'NIBP' in target_col:
            nibp_cols = [col for col in df.columns if 'NIBP' in col and col != target_col]
            exclude_cols.extend(nibp_cols)
        
        # 숫자형 컬럼만 선택
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        X = df[feature_cols]
        
        print(f"\n특성 개수: {len(feature_cols)}개")
        print(f"샘플 수: {len(X)}개")
        
        self.feature_names = feature_cols
        
        return X, y, feature_cols
    
    # ================================================================
    # 모델 학습 관련 메서드
    # ================================================================
    
    def train_models(self, X: pd.DataFrame, y: pd.Series, 
                    test_size: float = 0.2, random_state: int = 42,
                    models_to_train: Optional[List[str]] = None) -> Dict:
        """
        혈압 예측 모델 훈련 및 비교
        
        Args:
            X: 특성 데이터
            y: 타겟 데이터
            test_size: 테스트 데이터 비율
            random_state: 랜덤 시드
            models_to_train: 학습할 모델 리스트
        
        Returns:
            학습 결과 딕셔너리
        """
        print("=" * 70)
        print("🤖 모델 학습 시작")
        print("=" * 70)
        
        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"\n훈련 데이터: {len(X_train)}개")
        print(f"테스트 데이터: {len(X_test)}개")
        
        # 특성 스케일링
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 모델 정의
        all_models = {
            'Ridge': {
                'model': Ridge(alpha=1.0, random_state=random_state),
                'scaled': True,
                'description': '릿지 회귀 (L2 정규화)'
            },
            'RandomForest': {
                'model': RandomForestRegressor(
                    n_estimators=100, max_depth=15,
                    min_samples_split=5, min_samples_leaf=2,
                    random_state=random_state, n_jobs=-1
                ),
                'scaled': False,
                'description': '랜덤 포레스트'
            }
        }
        
        if models_to_train is None:
            models_to_train = ['Ridge', 'RandomForest']
        
        selected_models = {name: all_models[name] for name in models_to_train 
                          if name in all_models}
        
        results = {}
        
        for name, model_info in selected_models.items():
            print(f"\n{'='*60}")
            print(f"🔄 {name} 모델 훈련 중...")
            print(f"{'='*60}")
            
            model = model_info['model']
            use_scaled = model_info['scaled']
            
            X_tr = X_train_scaled if use_scaled else X_train
            X_te = X_test_scaled if use_scaled else X_test
            
            start_time = time.time()
            model.fit(X_tr, y_train)
            train_time = time.time() - start_time
            
            train_pred = model.predict(X_tr)
            test_pred = model.predict(X_te)
            
            train_mae = mean_absolute_error(y_train, train_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            train_r2 = r2_score(y_train, train_pred)
            
            test_mae = mean_absolute_error(y_test, test_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            test_r2 = r2_score(y_test, test_pred)
            
            results[name] = {
                'model': model,
                'scaled': use_scaled,
                'train_mae': train_mae,
                'train_rmse': train_rmse,
                'train_r2': train_r2,
                'test_mae': test_mae,
                'test_rmse': test_rmse,
                'test_r2': test_r2,
                'train_predictions': train_pred,
                'test_predictions': test_pred,
                'train_time': train_time
            }
            
            print(f"\n📊 성능 지표:")
            print(f"   테스트 MAE: {test_mae:.2f} mmHg")
            print(f"   테스트 RMSE: {test_rmse:.2f} mmHg")
            print(f"   테스트 R²: {test_r2:.3f}")
        
        # 최고 성능 모델 선택
        best_model_name = min(results.keys(), key=lambda x: results[x]['test_mae'])
        results['best_model_name'] = best_model_name
        
        print(f"\n🏆 최고 성능 모델: {best_model_name}")
        
        results.update({
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'scaler': self.scaler,
            'X_train_scaled': X_train_scaled,
            'X_test_scaled': X_test_scaled
        })
        
        self.models = results
        
        return results
    
    # ================================================================
    # 예측 관련 메서드
    # ================================================================
    
    def predict(self, X_new: Union[pd.DataFrame, Dict], 
               model_type: str = 'RandomForest') -> Dict:
        """
        새로운 데이터의 혈압 예측
        
        Args:
            X_new: 새로운 환자 데이터 (dict 또는 DataFrame)
            model_type: 사용할 모델 타입
        
        Returns:
            예측 결과 딕셔너리 (systolic_bp, diastolic_bp 포함)
        """
        # 입력 데이터 변환
        if isinstance(X_new, dict):
            X_new = pd.DataFrame([X_new])
        elif isinstance(X_new, pd.Series):
            X_new = pd.DataFrame([X_new])
        else:
            X_new = X_new.copy()
        
        # 모델이 학습되지 않은 경우 기본 예측 알고리즘 사용
        if not self.models or model_type not in self.models:
            return self._basic_prediction(X_new)
        
        # 특성 검증 및 선택
        try:
            X_pred_df = X_new[self.feature_names]
        except KeyError:
            # 필요한 특성이 없으면 기본 예측 사용
            return self._basic_prediction(X_new)
        
        # 모델 가져오기
        model = self.models[model_type]['model']
        use_scaled = self.models[model_type]['scaled']
        
        # 데이터 스케일링
        if use_scaled and self.scaler is not None:
            X_pred = self.scaler.transform(X_pred_df)
        else:
            X_pred = X_pred_df
        
        # 예측
        predictions = model.predict(X_pred)
        
        # 결과 반환 (Streamlit app과 호환되는 형식)
        # predictions는 NIBP_mean 또는 다른 혈압 값을 예측
        # 수축기/이완기로 변환
        systolic = predictions[0] if len(predictions) > 0 else 120
        diastolic = systolic * 0.67  # 대략적인 비율
        
        # RandomForest인 경우 신뢰 구간 계산
        confidence_interval = None
        if model_type == 'RandomForest' and hasattr(model, 'estimators_'):
            tree_predictions = np.array([tree.predict(X_pred) 
                                        for tree in model.estimators_])
            pred_std = np.std(tree_predictions, axis=0)
            
            confidence_interval = {
                'lower': predictions - 1.96 * pred_std,
                'upper': predictions + 1.96 * pred_std,
                'std': pred_std
            }
        
        results = {
            'systolic_bp': float(systolic),
            'diastolic_bp': float(diastolic),
            'predicted_bp': predictions,
            'model_used': model_type,
            'n_samples': len(predictions)
        }
        
        if confidence_interval is not None:
            results['confidence_interval'] = confidence_interval
        
        return results
    
    def _basic_prediction(self, patient_data: Union[pd.DataFrame, Dict]) -> Dict:
        """
        모델이 없을 때 사용하는 기본 예측 알고리즘
        
        Args:
            patient_data: 환자 데이터
        
        Returns:
            예측 결과 딕셔너리
        """
        # DataFrame을 dict로 변환
        if isinstance(patient_data, pd.DataFrame):
            if len(patient_data) > 0:
                patient_dict = patient_data.iloc[0].to_dict()
            else:
                patient_dict = {}
        else:
            patient_dict = patient_data
        
        # 기본값 설정
        age = patient_dict.get('age', 45)
        bmi = patient_dict.get('bmi', 23.0)
        smoking = patient_dict.get('smoking', 0)
        exercise_frequency = patient_dict.get('exercise_frequency', 2)
        stress_level = patient_dict.get('stress_level', 5)
        
        # 기본 예측 알고리즘
        base_systolic = 100 + (age * 0.5)
        base_diastolic = 60 + (age * 0.3)
        
        # 위험 요인 반영
        if bmi >= 30:
            base_systolic += 10
            base_diastolic += 5
        elif bmi >= 25:
            base_systolic += 5
            base_diastolic += 3
        
        if smoking == 1:
            base_systolic += 5
            base_diastolic += 3
        
        if exercise_frequency < 2:
            base_systolic += 3
            base_diastolic += 2
        
        if stress_level >= 7:
            base_systolic += 5
            base_diastolic += 3
        
        return {
            'systolic_bp': float(base_systolic),
            'diastolic_bp': float(base_diastolic),
            'model_used': '기본 알고리즘',
            'n_samples': 1
        }
    
    # ================================================================
    # 분석 관련 메서드
    # ================================================================
    
    def get_feature_importance(self, model_type: str = 'RandomForest',
                              top_n: int = 20) -> pd.DataFrame:
        """
        모델의 특성 중요도 추출
        
        Args:
            model_type: 모델 타입
            top_n: 표시할 상위 특성 개수
        
        Returns:
            특성 중요도 DataFrame
        """
        if model_type not in self.models:
            raise ValueError(f"모델 '{model_type}'를 찾을 수 없습니다.")
        
        model = self.models[model_type]['model']
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
        else:
            raise ValueError(f"{model_type} 모델은 특성 중요도를 제공하지 않습니다.")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    # ================================================================
    # Streamlit 지원 메서드
    # ================================================================
    
    def get_model_summary(self) -> Dict:
        """
        학습된 모델의 요약 정보 반환 (Streamlit 표시용)
        
        Returns:
            모델 요약 정보 딕셔너리
        """
        if not self.models:
            return {'status': 'no_models', 'message': '학습된 모델이 없습니다.'}
        
        best_name = self.models.get('best_model_name')
        if not best_name:
            return {'status': 'no_best_model', 'message': '최고 모델이 지정되지 않았습니다.'}
        
        best_result = self.models[best_name]
        
        summary = {
            'status': 'success',
            'best_model': best_name,
            'test_mae': best_result['test_mae'],
            'test_rmse': best_result['test_rmse'],
            'test_r2': best_result['test_r2'],
            'train_time': best_result.get('train_time', 0),
            'n_features': len(self.feature_names),
            'available_models': [k for k in self.models.keys() 
                               if k not in ['best_model_name', 'X_train', 'X_test', 
                                          'y_train', 'y_test', 'scaler', 
                                          'X_train_scaled', 'X_test_scaled']]
        }
        
        return summary
    
    def get_prediction_with_explanation(self, X_new: Union[pd.DataFrame, Dict],
                                       model_type: str = 'RandomForest',
                                       top_features: int = 5) -> Dict:
        """
        예측 결과와 설명 정보 반환 (Streamlit 표시용)
        
        Args:
            X_new: 새로운 환자 데이터
            model_type: 사용할 모델
            top_features: 표시할 주요 특성 개수
        
        Returns:
            예측 결과 및 설명 정보
        """
        # 기본 예측
        pred_result = self.predict(X_new, model_type)
        
        # 특성 중요도 추가
        try:
            importance_df = self.get_feature_importance(model_type, top_n=top_features)
            pred_result['top_features'] = importance_df.head(top_features).to_dict('records')
        except:
            pred_result['top_features'] = None
        
        # 입력 데이터의 주요 값 추가
        if isinstance(X_new, dict):
            X_df = pd.DataFrame([X_new])
        else:
            X_df = X_new
        
        pred_result['input_summary'] = {
            'n_features': len(X_df.columns),
            'sample_values': X_df.iloc[0].to_dict() if len(X_df) > 0 else {}
        }
        
        return pred_result
    
    def validate_input_data(self, X_new: Union[pd.DataFrame, Dict]) -> Dict:
        """
        입력 데이터 검증 (Streamlit에서 사용)
        
        Args:
            X_new: 검증할 데이터
        
        Returns:
            검증 결과 딕셔너리
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # DataFrame 변환
        if isinstance(X_new, dict):
            X_df = pd.DataFrame([X_new])
        elif isinstance(X_new, pd.Series):
            X_df = pd.DataFrame([X_new])
        else:
            X_df = X_new.copy()
        
        # 특성 이름 확인
        if not self.feature_names:
            result['valid'] = False
            result['errors'].append("모델이 학습되지 않았습니다. feature_names가 없습니다.")
            return result
        
        # 누락된 특성 확인
        missing_features = set(self.feature_names) - set(X_df.columns)
        if missing_features:
            result['valid'] = False
            result['errors'].append(f"누락된 특성: {list(missing_features)[:10]}")
        
        # 추가 특성 확인 (경고)
        extra_features = set(X_df.columns) - set(self.feature_names)
        if extra_features:
            result['warnings'].append(f"불필요한 특성이 있습니다 (무시됨): {list(extra_features)[:10]}")
        
        # 결측값 확인
        if X_df.isnull().any().any():
            null_cols = X_df.columns[X_df.isnull().any()].tolist()
            result['warnings'].append(f"결측값이 있는 컬럼: {null_cols[:10]}")
        
        # 무한대 값 확인
        numeric_cols = X_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if np.isinf(X_df[col]).any():
                result['warnings'].append(f"무한대 값이 있는 컬럼: {col}")
        
        return result
    
    def get_sample_data(self, n_samples: int = 5) -> pd.DataFrame:
        """
        테스트용 샘플 데이터 반환 (Streamlit 데모용)
        
        Args:
            n_samples: 반환할 샘플 수
        
        Returns:
            샘플 데이터 DataFrame
        """
        if 'X_test' not in self.models:
            raise ValueError("학습된 모델이 없습니다. 먼저 모델을 학습하세요.")
        
        X_test = self.models['X_test']
        return X_test.head(n_samples)
    
    def plot_prediction_results(self, model_type: str = 'RandomForest',
                               save_path: Optional[str] = None):
        """
        예측 결과 시각화 (Streamlit용)
        
        Args:
            model_type: 시각화할 모델
            save_path: 저장 경로 (None이면 표시만)
        
        Returns:
            matplotlib figure 객체
        """
        if model_type not in self.models:
            raise ValueError(f"모델 '{model_type}'를 찾을 수 없습니다.")
        
        y_test = self.models['y_test']
        y_pred = self.models[model_type]['test_predictions']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 실제값 vs 예측값
        axes[0, 0].scatter(y_test, y_pred, alpha=0.5, s=50)
        axes[0, 0].plot([y_test.min(), y_test.max()], 
                        [y_test.min(), y_test.max()], 
                        'r--', lw=2, label='Perfect')
        axes[0, 0].set_xlabel('Actual BP (mmHg)', fontsize=12)
        axes[0, 0].set_ylabel('Predicted BP (mmHg)', fontsize=12)
        axes[0, 0].set_title(f'Actual vs Predicted - {model_type}', 
                            fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 잔차 플롯
        residuals = y_test - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.5, s=50)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('Predicted BP (mmHg)', fontsize=12)
        axes[0, 1].set_ylabel('Residuals (mmHg)', fontsize=12)
        axes[0, 1].set_title('Residual Plot', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 잔차 분포
        axes[1, 0].hist(residuals, bins=30, edgecolor='black', 
                       alpha=0.7, color='skyblue')
        axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1, 0].set_xlabel('Residuals (mmHg)', fontsize=12)
        axes[1, 0].set_ylabel('Frequency', fontsize=12)
        axes[1, 0].set_title('Residual Distribution', 
                            fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 4. 성능 지표
        mae = self.models[model_type]['test_mae']
        rmse = self.models[model_type]['test_rmse']
        r2 = self.models[model_type]['test_r2']
        
        metrics_text = f"Performance Metrics\n\n"
        metrics_text += f"MAE:  {mae:.2f} mmHg\n"
        metrics_text += f"RMSE: {rmse:.2f} mmHg\n"
        metrics_text += f"R²:   {r2:.3f}\n\n"
        metrics_text += f"Residual Stats:\n"
        metrics_text += f"Mean: {residuals.mean():.2f}\n"
        metrics_text += f"Std:  {residuals.std():.2f}"
        
        axes[1, 1].text(0.5, 0.5, metrics_text, 
                       ha='center', va='center',
                       fontsize=14, family='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 그래프 저장: {save_path}")
        
        return fig
    
    def plot_feature_importance(self, model_type: str = 'RandomForest',
                               top_n: int = 15,
                               save_path: Optional[str] = None):
        """
        특성 중요도 시각화 (Streamlit용)
        
        Args:
            model_type: 시각화할 모델
            top_n: 표시할 상위 특성 개수
            save_path: 저장 경로
        
        Returns:
            matplotlib figure 객체
        """
        importance_df = self.get_feature_importance(model_type, top_n=len(self.feature_names))
        top_features = importance_df.head(top_n)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. 상위 특성 바 차트
        axes[0].barh(range(len(top_features)), top_features['importance'], 
                    color='steelblue', edgecolor='black', alpha=0.7)
        axes[0].set_yticks(range(len(top_features)))
        axes[0].set_yticklabels(top_features['feature'])
        axes[0].set_xlabel('Importance', fontsize=12)
        axes[0].set_title(f'Top {top_n} Features - {model_type}', 
                         fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='x')
        axes[0].invert_yaxis()
        
        # 2. 누적 중요도
        cumsum = importance_df['importance'].cumsum()
        cumsum_pct = (cumsum / cumsum.iloc[-1]) * 100
        
        axes[1].plot(range(1, len(cumsum_pct) + 1), cumsum_pct, 
                    marker='o', linewidth=2, markersize=4)
        axes[1].axhline(y=80, color='r', linestyle='--', 
                       linewidth=2, label='80%')
        axes[1].axhline(y=90, color='orange', linestyle='--', 
                       linewidth=2, label='90%')
        axes[1].set_xlabel('Number of Features', fontsize=12)
        axes[1].set_ylabel('Cumulative Importance (%)', fontsize=12)
        axes[1].set_title('Cumulative Importance', 
                         fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 그래프 저장: {save_path}")
        
        return fig

    # ================================================================
    # 통합 파이프라인 메서드
    # ================================================================
    
    def save_model(self, model_type: str, filepath: str):
        """모델 저장"""
        if model_type not in self.models:
            raise ValueError(f"모델 '{model_type}'를 찾을 수 없습니다.")
        
        model = self.models[model_type]['model']
        joblib.dump(model, filepath)
        print(f"✅ 모델 저장: {filepath}")
    
    def save_scaler(self, filepath: str):
        """스케일러 저장"""
        if self.scaler is None:
            raise ValueError("저장할 스케일러가 없습니다.")
        
        joblib.dump(self.scaler, filepath)
        print(f"✅ 스케일러 저장: {filepath}")
    
    def save_metadata(self, filepath: str, target_column: str = 'NIBP_mean'):
        """메타데이터 저장"""
        best_name = self.models.get('best_model_name')
        
        if best_name is None:
            raise ValueError("학습된 모델이 없습니다.")
        
        metadata = {
            'model_name': best_name,
            'test_mae': float(self.models[best_name]['test_mae']),
            'test_rmse': float(self.models[best_name]['test_rmse']),
            'test_r2': float(self.models[best_name]['test_r2']),
            'n_features': len(self.feature_names),
            'features': self.feature_names,
            'target_column': target_column,
            'train_date': str(pd.Timestamp.now())
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 메타데이터 저장: {filepath}")
    
    def load_model(self, model_type: str, filepath: str):
        """모델 로드"""
        model = joblib.load(filepath)
        if model_type not in self.models:
            self.models[model_type] = {}
        self.models[model_type]['model'] = model
        print(f"✅ 모델 로드: {filepath}")
    
    def load_scaler(self, filepath: str):
        """스케일러 로드"""
        self.scaler = joblib.load(filepath)
        print(f"✅ 스케일러 로드: {filepath}")
    
    # ================================================================
    # 통합 파이프라인 메서드
    # ================================================================
    
    def full_pipeline(self, max_records: Optional[int] = 100,
                     target_col: str = 'NIBP_mean',
                     test_size: float = 0.2) -> Dict:
        """
        전체 파이프라인 실행
        
        Args:
            max_records: 로드할 최대 레코드 수
            target_col: 타겟 변수
            test_size: 테스트 데이터 비율
        
        Returns:
            학습 결과
        """
        print("=" * 80)
        print("🩺 PhysioNet 혈압 예측 AI 시스템")
        print("=" * 80)
        
        # 1. 데이터 로드
        df = self.load_all_patient_data(max_records=max_records)
        
        # 2. 전처리
        df_clean = self.preprocess_data(df, target_column=target_col)
        
        # 3. 특성 준비
        X, y, features = self.prepare_features(df_clean, target_col=target_col)
        
        # 4. 모델 학습
        results = self.train_models(X, y, test_size=test_size)
        
        print("\n✅ 파이프라인 완료!")
        
        return results



