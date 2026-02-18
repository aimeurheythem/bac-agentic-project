import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Loader2 } from 'lucide-react';
import { useAppStore } from '../../store/appStore';
import { fetchStreams, fetchStreamSpecialties } from '../../lib/api';
import type { Stream, SpecialtyOption } from '../../types';
import './StreamSelection.css';

export default function StreamSelection() {
  const navigate = useNavigate();
  const [streams, setStreams] = useState<Stream[]>([]);
  const [specialties, setSpecialties] = useState<SpecialtyOption[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  const {
    selectedStream,
    selectedSpecialty,
    setSelectedStream,
    setSelectedSpecialty,
    setAvailableStreams,
    completeOnboarding,
  } = useAppStore();

  useEffect(() => { loadStreams(); }, []);

  useEffect(() => {
    if (selectedStream?.hasOptions) {
      loadSpecialties(selectedStream.id);
    } else {
      setSpecialties([]);
    }
  }, [selectedStream]);

  const loadStreams = async () => {
    try {
      const data = await fetchStreams();
      setStreams(data);
      setAvailableStreams(data);
    } catch (e) {
      console.error('Failed to load streams:', e);
    } finally {
      setLoading(false);
    }
  };

  const loadSpecialties = async (streamId: number) => {
    try {
      const data = await fetchStreamSpecialties(streamId);
      setSpecialties(data);
    } catch (e) {
      console.error('Failed to load specialties:', e);
    }
  };

  const handleStreamSelect = (stream: Stream) => {
    if (selectedStream?.id === stream.id) {
      setSelectedStream(null);
      setSelectedSpecialty(null);
    } else {
      setSelectedStream(stream);
      setSelectedSpecialty(null);
    }
  };

  const handleSpecialtySelect = (specialty: SpecialtyOption) => {
    setSelectedSpecialty(selectedSpecialty?.code === specialty.code ? null : specialty);
  };

  const handleContinue = async () => {
    if (!selectedStream) return;
    if (selectedStream.hasOptions && !selectedSpecialty) return;
    setSaving(true);
    try {
      completeOnboarding();
      navigate('/dashboard');
    } catch (e) {
      console.error('Failed:', e);
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="ob-loading">
        <Loader2 className="ob-spinner" />
      </div>
    );
  }

  const canContinue = selectedStream && (!selectedStream.hasOptions || selectedSpecialty);

  return (
    <div className="ob-container">

      {/* Header — identical to dashboard header */}
      <header className="ob-header">
        <div className="ob-header-content">
          <div className="ob-brand">
            <span className="ob-brand-dot" />
            <span className="ob-brand-name">Bac Agent</span>
          </div>
          <div className="ob-step-label">الإعداد الأولي</div>
        </div>
      </header>

      {/* Body */}
      <main className="ob-main">
        <div className="ob-card">

          {/* Title block */}
          <div className="ob-title-block">
            <p className="ob-eyebrow">خطوة 1 من 1</p>
            <h1 className="ob-title">اختر شعبتك</h1>
            <p className="ob-subtitle">يحدد الاختيار المواد والمعاملات ونمط الإجابة</p>
          </div>

          {/* Stream chips */}
          <div className="ob-chips-row">
            {streams.map((stream) => {
              const isSelected = selectedStream?.id === stream.id;
              return (
                <button
                  key={stream.id}
                  onClick={() => handleStreamSelect(stream)}
                  className={`ob-chip${isSelected ? ' ob-chip--active' : ''}`}
                >
                  {stream.nameAr || stream.name}
                </button>
              );
            })}
          </div>

          {/* Specialty chips (Technique Math only) */}
          {selectedStream?.hasOptions && specialties.length > 0 && (
            <div className="ob-specialty-block">
              <p className="ob-specialty-label">اختر التخصص</p>
              <div className="ob-chips-row">
                {specialties.map((sp) => {
                  const isSelected = selectedSpecialty?.code === sp.code;
                  return (
                    <button
                      key={sp.code}
                      onClick={() => handleSpecialtySelect(sp)}
                      className={`ob-chip ob-chip--specialty${isSelected ? ' ob-chip--active' : ''}`}
                    >
                      {sp.nameAr || sp.name}
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          {/* Status line */}
          {selectedStream && (
            <p className="ob-selection-status">
              <span className="ob-status-dot" />
              {selectedStream.nameAr || selectedStream.name}
              {selectedSpecialty && <> · {selectedSpecialty.nameAr || selectedSpecialty.name}</>}
            </p>
          )}

          {/* Continue */}
          <div className="ob-actions">
            {!canContinue && selectedStream?.hasOptions && (
              <span className="ob-hint">يرجى اختيار التخصص أولاً</span>
            )}
            <button
              onClick={handleContinue}
              disabled={!canContinue || saving}
              className={`ob-continue${canContinue ? '' : ' ob-continue--disabled'}`}
            >
              {saving
                ? <Loader2 size={16} className="ob-btn-spinner" />
                : <ArrowLeft size={16} />
              }
              {saving ? 'جارٍ الحفظ…' : 'ابدأ الآن'}
            </button>
          </div>

        </div>
      </main>
    </div>
  );
}
