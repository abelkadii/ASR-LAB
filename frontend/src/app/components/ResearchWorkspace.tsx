import { ModelsWorkspace } from './workspaces/ModelsWorkspace';
import { DatasetWorkspace } from './workspaces/DatasetWorkspace';
import { TranscribeResearchWorkspace } from './workspaces/TranscribeResearchWorkspace';
import { LabelResearchWorkspace } from './workspaces/LabelResearchWorkspace';
import { BatchResearchWorkspace } from './workspaces/BatchResearchWorkspace';
import { HomeWorkspace } from './workspaces/HomeWorkspace';
import { Workspace } from '../api';

interface ResearchWorkspaceProps {
  workspace: Workspace;
  onWorkspaceChange: (workspace: Workspace) => void;
}

export function ResearchWorkspace({ workspace, onWorkspaceChange }: ResearchWorkspaceProps) {
  return (
    <div className="flex-1 overflow-hidden">
      {workspace === 'home' && <HomeWorkspace onWorkspaceChange={onWorkspaceChange} />}
      {workspace === 'workbench' && <TranscribeResearchWorkspace onWorkspaceChange={onWorkspaceChange} />}
      {workspace === 'models' && <ModelsWorkspace onWorkspaceChange={onWorkspaceChange} />}
      {workspace === 'dataset' && <DatasetWorkspace onWorkspaceChange={onWorkspaceChange} />}
      {workspace === 'label' && <LabelResearchWorkspace />}
      {workspace === 'analysis' && <BatchResearchWorkspace onWorkspaceChange={onWorkspaceChange} />}
    </div>
  );
}
