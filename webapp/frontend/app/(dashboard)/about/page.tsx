import { Typography } from "@mui/material";
import { PageContainer } from "@toolpad/core";

export default function SearchPage() {
  return (
    <PageContainer breadcrumbs={[]}>
    <Typography>
    Project 001 is a project started by the <a href="https://www.educationfoundationstlucie.org/"> Saint Lucie County Education Foundation
	</a>
	and lead by board member&nbsp;
	<a href="https://www.linkedin.com/in/jameseabbott/">
		James Abbott
	</a>
	. <br/>
	Along with students from Saint Lucie County high schools, the team is working to solve traffic problems around the county using AI and Machine Learning.
	<br />
	<br />
	<strong>
	Team
	</strong>

	<br/>
	The team is made up of high school students in Saint Lucie County.

    <br/> <br/>

	<strong>Resources:</strong>
    <br/><br/>

	Project 001 initial video:&nbsp;
	<a href="https://youtu.be/WCSUJPJYDQI">
		Project 001 - Using AI to control traffic signals
	</a>
    </Typography>

    </PageContainer>
  );
}
