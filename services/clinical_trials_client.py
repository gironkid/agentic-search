"""
Real ClinicalTrials.gov API integration for searching clinical trials.
"""

import aiohttp
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ClinicalTrialsClient:
    """
    Client for interacting with ClinicalTrials.gov API v2.
    """
    
    BASE_URL = "https://clinicaltrials.gov/api/v2"
    
    def __init__(self):
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def search(
        self,
        condition: str,
        intervention: Optional[str] = None,
        status: Optional[str] = None,
        phase: Optional[List[str]] = None,
        max_results: int = 10
    ) -> Dict[str, Any]:
        """
        Search for clinical trials.
        
        Args:
            condition: Medical condition
            intervention: Treatment or intervention
            status: Trial status (recruiting, active, completed, all)
            phase: List of phases (Phase 1, Phase 2, Phase 3, Phase 4)
            max_results: Maximum results to return
            
        Returns:
            Dictionary with trial results
        """
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        try:
            # Build query filter
            query_parts = []
            
            # Add condition
            if condition:
                query_parts.append(f"AREA[Condition]{condition}")
            
            # Add intervention
            if intervention:
                query_parts.append(f"AREA[Intervention]{intervention}")
            
            # Add status filter
            if status and status != "all":
                status_map = {
                    "recruiting": "RECRUITING",
                    "active": "ACTIVE_NOT_RECRUITING",
                    "completed": "COMPLETED"
                }
                if status.lower() in status_map:
                    query_parts.append(f"AREA[OverallStatus]{status_map[status.lower()]}")
            
            # Add phase filter
            if phase:
                phase_str = " OR ".join([f"AREA[Phase]{p}" for p in phase])
                query_parts.append(f"({phase_str})")
            
            # Combine query parts
            query = " AND ".join(query_parts) if query_parts else condition
            
            # API parameters
            params = {
                "query.cond": condition,
                "pageSize": min(max_results, 100),  # API max is 100
                "format": "json"
            }
            
            if intervention:
                params["query.intr"] = intervention
            
            # Make API request
            url = f"{self.BASE_URL}/studies"
            async with self.session.get(url, params=params) as response:
                data = await response.json()
            
            # Parse results
            studies = data.get("studies", [])
            trials = []
            
            for study in studies[:max_results]:
                protocol = study.get("protocolSection", {})
                id_module = protocol.get("identificationModule", {})
                status_module = protocol.get("statusModule", {})
                design_module = protocol.get("designModule", {})
                arms_module = protocol.get("armsInterventionsModule", {})
                
                # Extract key information
                trial = {
                    "nct_id": id_module.get("nctId", ""),
                    "title": id_module.get("briefTitle", ""),
                    "status": status_module.get("overallStatus", ""),
                    "phase": ", ".join(design_module.get("phases", [])),
                    "start_date": status_module.get("startDateStruct", {}).get("date", ""),
                    "completion_date": status_module.get("completionDateStruct", {}).get("date", ""),
                    "enrollment": design_module.get("enrollmentInfo", {}).get("count", 0),
                    "study_type": design_module.get("studyType", ""),
                    "interventions": [],
                    "conditions": protocol.get("conditionsModule", {}).get("conditions", []),
                    "locations": [],
                    "url": f"https://clinicaltrials.gov/study/{id_module.get('nctId', '')}"
                }
                
                # Extract interventions
                for intervention in arms_module.get("interventions", []):
                    trial["interventions"].append({
                        "type": intervention.get("type", ""),
                        "name": intervention.get("name", "")
                    })
                
                # Extract locations (first few)
                locations_module = protocol.get("contactsLocationsModule", {})
                for location in locations_module.get("locations", [])[:3]:
                    trial["locations"].append({
                        "facility": location.get("facility", ""),
                        "city": location.get("city", ""),
                        "country": location.get("country", "")
                    })
                
                trials.append(trial)
            
            return {
                "condition": condition,
                "total_trials": len(studies),
                "returned": len(trials),
                "trials": trials
            }
            
        except Exception as e:
            logger.error(f"ClinicalTrials.gov search error: {e}")
            return {
                "condition": condition,
                "error": str(e),
                "trials": []
            }


async def test_clinical_trials():
    """Test the ClinicalTrials.gov client"""
    
    print("\n" + "="*60)
    print("TESTING CLINICAL TRIALS API")
    print("="*60)
    
    async with ClinicalTrialsClient() as client:
        print("\n[TEST] Searching for: biliary atresia")
        results = await client.search(
            condition="biliary atresia",
            status="recruiting",
            max_results=3
        )
        
        print(f"Total trials found: {results.get('total_trials', 0)}")
        print(f"Trials returned: {results.get('returned', 0)}")
        
        for i, trial in enumerate(results.get("trials", []), 1):
            print(f"\n{i}. {trial.get('title', 'No title')}")
            print(f"   NCT: {trial.get('nct_id', 'N/A')}")
            print(f"   Status: {trial.get('status', 'Unknown')}")
            print(f"   Phase: {trial.get('phase', 'N/A')}")
            print(f"   URL: {trial.get('url', 'N/A')}")
            
            if trial.get('interventions'):
                print(f"   Interventions:")
                for inv in trial['interventions'][:2]:
                    print(f"     - {inv.get('type', '')}: {inv.get('name', '')}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_clinical_trials())