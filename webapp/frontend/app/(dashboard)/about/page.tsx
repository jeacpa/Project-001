import { Typography } from "@mui/material";
import { PageContainer } from "@toolpad/core";

export default function SearchPage() {
  return (
    <PageContainer breadcrumbs={[]}>
    <Typography>
    <strong> Project 001 - "If you want to be successful, solve a problem" </strong>
    <br /><br/>
    Project 001 is a project started by the
    <a href="https://www.educationfoundationstlucie.org/">&nbsp;Saint Lucie County Education Foundation</a>
    &nbsp;and lead by board member&nbsp; <a href="https://www.linkedin.com/in/jameseabbott/"> James Abbott</a>.
    <br/>
	Along with students from Saint Lucie County high schools, the team is working to solve traffic problems
	around the county using AI and Machine Learning.  Eventually, we would like to share our solution throughout
	Saint Lucie County, the rest of Florida, and to other states.
	<br />

	<br />
	<strong>
		 Team:
	</strong>

	<br/>
	The team is made up of high school students in Saint Lucie County.

    <br/> <br/>

	<strong>Resources:</strong>
    <br/>

	Project 001 initial video:&nbsp;
	<a href="https://youtu.be/WCSUJPJYDQI"> Project 001 - Using AI to control traffic signals </a>
	<br/>
	<br/>
	<strong> Contact us: </strong>
    <br/>

	If you are interested in helping with Project 001 or any other of our projects, please contact
	<a href="mailto:thom.jones@educationfoundationstlucie.org" subject="Education Foundation - Project 001"> Thom Jones</a>
	&nbsp; or <a href="mailto:jabbott@efslc.org" subject="Education Foundation - Project 001"> James Abbott</a>

    </Typography>

    </PageContainer>
  );
}
